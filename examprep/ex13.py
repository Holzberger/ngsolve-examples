from ngsolve import *
from netgen.geom2d import SplineGeometry
import matplotlib.pyplot as plt
import numpy as np
import time

def mesh_cylinder():
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
    geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    mesh = Mesh( geo.GenerateMesh(maxh=0.05))
    mesh.Curve(3)
    return mesh


def SolveStokes_cyl(VxQ, nu=1):
    
    X = FESpace(VxQ)
    if len(VxQ)==3:
        (u,p,lam),(v,q,mu) = X.TnT()
    else:
        (u,p),(v,q) = X.TnT()

    a = BilinearForm(X)
    def eps(u): 
        return 1/2*(grad(u) + grad(u).trans)
    a += (2*nu*InnerProduct(eps(u), eps(v)))*dx
    a += (-div(u)*q - div(v)*p)*dx 
    
    if len(VxQ)==3:
        a += (lam*q + mu*p)*dx
    
    a.Assemble()
  
     
    gfu = GridFunction(X)
   
    # add u with BC but f=0
    uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
    
    res = gfu.vec.CreateVector()
    res.data = -a.mat * gfu.vec


    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data += inv * res
    
    Draw(gfu.components[0], mesh, "vel")
    Draw(gfu.components[1], mesh, "pressure")
    Draw(Norm(gfu.components[0]), mesh, "|vel|")
    SetVisualization(max=2)


    V2 = VectorL2(mesh, 1)
    
    res_u = GridFunction(V2)
    res_u.Set(gfu.components[1].Deriv())
    Draw(res_u, mesh, "residual")

    return gfu

def mesh_rectangle(hmax=0.1):
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (1, 1), bcs = ("wall", "wall", "wall", "wall"))
    mesh = Mesh( geo.GenerateMesh(maxh=hmax))
    return mesh


def SolveStokes_rec(VxQ, nu=1, k=1):
    
    X = FESpace(VxQ)
    if len(VxQ)==3:
        (u,p,lam),(v,q,mu) = X.TnT()
    else:
        (u,p),(v,q) = X.TnT()

    a = BilinearForm(X)
    def eps(u): 
        return 1/2*(grad(u) + grad(u).trans)
    a += nu*InnerProduct(eps(u), eps(v))*dx
    a += (-div(u)*q - div(v)*p)*dx 
    
    if len(VxQ)==3:
        a += (lam*q + mu*p)*dx
    else:
        a+= (-k*p*q)*dx
    
    a.Assemble()


    psi = (x*(x-1)*y*(y-1))**2
    p_ex = x**5 + y**5 - 1/3
    u_ex = CoefficientFunction( (psi.Diff(y), -psi.Diff(x)) )
    sol_ex = GridFunction(X)
    sol_ex.components[0].Set(u_ex)
    sol_ex.components[1].Set(p_ex)


    grad_ue = CoefficientFunction( (u_ex[0].Diff(x),
                                    u_ex[0].Diff(y),
                                    u_ex[1].Diff(x),
                                    u_ex[1].Diff(y)),
                                    dims=(2,2))
    eps_nu_uex = -1/2*nu*(grad_ue + grad_ue.trans)
    f = CoefficientFunction((   eps_nu_uex[0,0].Diff(x)+eps_nu_uex[0,1].Diff(y),
                                eps_nu_uex[1,0].Diff(x)+eps_nu_uex[1,1].Diff(y)))
    f += CoefficientFunction((p_ex.Diff(x), p_ex.Diff(y)))
    #Draw(f, mesh, "f")


    rhs = LinearForm(X)
    rhs += f*v*dx
    rhs.Assemble()


    gfu = GridFunction(X)
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="pardiso")
    gfu.vec.data = inv * rhs.vec
    
    #Draw(gfu.components[0], mesh, "vel")
    #Draw(gfu.components[1], mesh, "pressure")
    #Draw(Norm(gfu.components[0]), mesh, "|vel|")

    return gfu, sol_ex
   

def get_FEspace(dir_bc, order=[2,1], conti=[1,1], bubble=0, lag_multi=True):
    VxQ = []
    
    V = VectorH1(mesh, order=order[0], dirichlet = dir_bc)
    if bubble>0:
        V.SetOrder(TRIG, bubble)
        V.Update()
    VxQ.append(V)

    if conti[1]==1:
        Q = H1(mesh, order=order[1])
    elif conti[1]==0:
        Q = L2(mesh, order=order[1])
    VxQ.append(Q)
    
    if lag_multi: # lag. multi to enfore zero mean value constraint
        N = NumberSpace(mesh)
        VxQ.append(N)

    return VxQ


def get_norms(gfu, sol_ex):
    p_norm  = Integrate((sol_ex.components[1]-gfu.components[1])**2,mesh)
    delta_u = sol_ex.components[0]-gfu.components[0]
    delta_du = grad(sol_ex.components[0])-grad(gfu.components[0])
    u_norm  = Integrate(InnerProduct(delta_u,delta_u),mesh)
    du_norm = Integrate(InnerProduct(delta_du,delta_du),mesh)
    p_L2  = p_norm**0.5
    u_L2  = u_norm**0.5
    u_H1s = du_norm**0.5
    return p_L2, u_L2, u_H1s 



H_max    = np.array([1, 1/2, 1/4, 1/8, 1/16])

elements = [ #[[2,1], [1,1], 0, True, "TH21"],
             #[[3,2], [1,1], 0, True, "TH32"],
             [[1,1], [1,1], 3, True, "MINI"],
             [[2,0], [1,0], 0, True, "P2P0"],
             [[2,1], [1,0], 3, True, "P2^+"],
             ] 


P_L2 = []
U_L2 = []
U_H1s = []

for elem in elements:
    P_L2.append([])
    U_L2.append([])
    U_H1s.append([])
    
    for h_max in H_max:
        print("hmax = ",h_max)
        geom = 1

        if geom ==0:
            mesh = mesh_cylinder()
            dir_bc = "wall|inlet|cyl" 

        if geom ==1:
            mesh = mesh_rectangle(h_max)
            dir_bc = "wall" 


        VxQ = get_FEspace(dir_bc, order=elem[0], conti=elem[1], bubble=elem[2], lag_multi=elem[3])

        gfu, sol_ex = SolveStokes_rec(VxQ, 1e-6, 1e-4)

        p_L2, u_L2, u_H1s = get_norms(gfu, sol_ex)
        P_L2[-1].append(p_L2)
        U_L2[-1].append(u_L2)
        U_H1s[-1].append(u_H1s)
        print(u_L2)

#plt.ioff() 
fig, ax = plt.subplots()
for n, elem in enumerate(elements):
    #ax.loglog(H_max, P_L2[n],".-",label="p L2 "+elem[-1])
    #ax.loglog(H_max, U_L2[n],".-",label="u L2 "+elem[-1])
    ax.loglog(H_max, U_H1s[n],".-",label="u H1 sem "+elem[-1])
ax.loglog(H_max, H_max,":",label="O(h)",color="black")
ax.loglog(H_max, H_max**2,":",label="O(h^2)",color="black")
ax.loglog(H_max, H_max**3,":",label="O(h^3)",color="black")


ax.legend(loc='upper left',fontsize=12)
ax.grid(True, which="both",linewidth=0.5)
ax.set_xlabel('$h_{max}$',fontsize=14)
ax.set_ylabel("error",fontsize=14)

#plt.savefig("norms.pdf")



