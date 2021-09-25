from ngsolve import *
import netgen.gui

from netgen.geom2d import SplineGeometry
import matplotlib.pyplot as plt
import numpy as np
##Setup Geometry
def MakeGeometry(hmax):
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (1, 1), bcs = ("wall", "wall", "wall", "wall"))
    mesh = Mesh( geo.GenerateMesh(maxh=hmax))
    Draw(mesh)
    return mesh

#SolveStokes
def SolveStokes(X,nu,alpha,h,mesh,consistent=True):
    ##Test and Trialfunctions
    (u,p,lam),(v,q,mu) = X.TnT()
   
    eps_u = 1/2*(grad(u)+grad(u).trans)
    eps_v = 1/2*(grad(v)+grad(v).trans)

    ##Initializes solution and rhs
    psi = (x*(x-1)*y*(y-1))**2
    pres = x**5 + y**5 - 1/3
    u_xe = psi.Diff(y)
    u_ye = -psi.Diff(x)
    exact_sol = GridFunction(X)
    exact_sol.components[0].Set((u_xe,u_ye))
    exact_sol.components[1].Set(pres)

    grad_ue = CoefficientFunction((u_xe.Diff(x),u_xe.Diff(y),u_ye.Diff(x),u_ye.Diff(y)),dims=(2,2))
    eps_nu_ue = -nu*1/2*(grad_ue + grad_ue.trans)
    f = CoefficientFunction((eps_nu_ue[0,0].Diff(x)+eps_nu_ue[0,1].Diff(y),eps_nu_ue[1,0].Diff(x)+eps_nu_ue[1,1].Diff(y)))
    f+= CoefficientFunction((pres.Diff(x),pres.Diff(y)))
    Draw(f, mesh, "f")

    Hesse = u.Operator("hesse")
    div_eps_u =CoefficientFunction((Hesse[0,0]+0.5*(Hesse[0,2]+Hesse[1,1]),Hesse[1,3]+0.5*(Hesse[1,0]+Hesse[0,1])))
    Hesse = v.Operator("hesse")
    div_eps_v = CoefficientFunction((Hesse[0,0]+0.5*(Hesse[0,2]+Hesse[1,1]),Hesse[1,3]+0.5*(Hesse[1,0]+Hesse[0,1])))
    ##Assemble BLF
    a = BilinearForm(X)
    a += (InnerProduct(eps_u,eps_v)-div(u)*q-div(v)*p+lam*q+mu*p)*dx
    ##Add Stabilization
    a += -alpha*h*h*(InnerProduct(-nu*div_eps_u+grad(p),-nu*div_eps_v+grad(q)))*dx
    a.Assemble()

    ##Assemble LF
    rhs = LinearForm(X)
    rhs += f*v*dx
    ##Add consistency term
    if consistent:
        rhs += -alpha*h*h*InnerProduct(f,-nu*div_eps_v+grad(q))*dx
    rhs.Assemble()

    ##Solve System
    gfu = GridFunction(X)
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="sparsecholesky")
    gfu.vec.data = inv*rhs.vec

    return gfu, exact_sol

## Setup ##
index = 6
alpha = 1e-6
nu = 1
hmax = np.exp2(-np.arange(1,index))
norm_u = np.zeros(index-1)
norm_du = np.zeros(index-1)
norm_p = np.zeros(index-1)

#Mini H1-P1-bubble, H1-P1
#P2P0 H1-P2, L2-const
#P2-bubble H1-P2-bubble, L2-P1

for i in range(index-1):
    mesh = MakeGeometry(hmax[i])
    V = VectorH1(mesh, order=2, dirichlet="wall")
    #V.SetOrder(TRIG,3)
    Q = H1(mesh, order=2)
    N = NumberSpace(mesh)
    X = FESpace([V,Q,N])

    ##Solve System
    gfu, sol = SolveStokes(X,nu,alpha,hmax[i],mesh,True)
    delta_u = sol.components[0]-gfu.components[0]
    delta_du = grad(sol.components[0])-grad(gfu.components[0])
    ##Calculate norms
    p_norm = Integrate((sol.components[1]-gfu.components[1])**2,mesh)
    u_norm = Integrate(InnerProduct(delta_u,delta_u),mesh)
    du_norm = Integrate(InnerProduct(delta_du,delta_du),mesh)

    norm_p[i] = np.sqrt(p_norm)
    norm_u[i] = np.sqrt(u_norm)
    norm_du[i] = np.sqrt(du_norm)

## Draw stuff
# Draw(gfu.components[0], mesh, "fem_vel")
# Draw(gfu.components[1], mesh, "fem_pres")
# Draw(sol.components[0], mesh, "exact_vel")
# Draw(sol.components[1], mesh, "exact_pres")

fig, ax = plt.subplots()
ax.loglog(hmax,norm_p,marker='o',label='$||p||_{L_2}$')
ax.loglog(hmax,norm_u,marker='o',label='$||u||_{L_2}$')
ax.loglog(hmax,norm_du,marker='o',label='$|u|_{H_1}$')
ax.loglog(hmax,hmax)
ax.loglog(hmax,np.power(hmax,2))
ax.grid(True, which="both",linewidth=0.5)
ax.set_xlabel('$h_{max}$',fontsize=14)
ax.set_ylabel("error",fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc='upper left',fontsize=12)
plt.show()
