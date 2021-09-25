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
def SolveStokes(X,nu,lagrange_mult=True,k=1):
    ##Test and Trialfunctions
    if lagrange_mult:
        (u,p,lam),(v,q,mu) = X.TnT()
    else:
        (u,p),(v,q) = X.TnT()

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

    ##Assemble BLF
    a = BilinearForm(X)
    if lagrange_mult:
        a += (InnerProduct(eps_u,eps_v)-div(u)*q-div(v)*p+lam*q+mu*p)*dx
    else:
        a += (InnerProduct(eps_u,eps_v)-div(u)*q-div(v)*p-k*p*q)*dx
    a.Assemble()

    ##Assemble LF
    rhs = LinearForm(X)
    rhs += f*v*dx
    rhs.Assemble()

    ##Solve System
    gfu = GridFunction(X)
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec

    return gfu, exact_sol

## Setup ##
index = 6
nu = 1
lagrange_mult = False
hmax = np.exp2(-np.arange(0,index))
norm_u = np.zeros(index)
norm_du = np.zeros(index)
norm_p = np.zeros(index)

#Mini H1-P1-bubble, H1-P1
#P2P0 H1-P2, L2-const
#P2-bubble H1-P2-bubble, L2-P1
for i in range(index):
    mesh = MakeGeometry(hmax[i])
    V = VectorH1(mesh, order=2, dirichlet="wall")
    #V.SetOrder(TRIG,3)
    Q = L2(mesh, order=0)
    if lagrange_mult:
        N = NumberSpace(mesh)
        X = FESpace([V,Q,N])
    else:
        X = FESpace([V,Q])

    ##Solve System
    gfu, sol = SolveStokes(X,nu,lagrange_mult,1e-7)
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
Draw(gfu.components[0], mesh, "fem_vel")
Draw(gfu.components[1], mesh, "fem_pres")
Draw(sol.components[0], mesh, "exact_vel")
Draw(sol.components[1], mesh, "exact_pres")

#fig, ax = plt.subplots()
#ax.loglog(hmax,norm_p,marker='o',label='$||p||_{L_2}$')
#ax.loglog(hmax,norm_u,marker='o',label='$||u||_{L_2}$')
#ax.loglog(hmax,norm_du,marker='o',label='$|u|_{H_1}$')
#ax.loglog(hmax,hmax)
#ax.loglog(hmax,np.power(hmax,2))
#ax.grid(True, which="both",linewidth=0.5)
#ax.set_xlabel('$h_{max}$',fontsize=14)
#ax.set_ylabel("error",fontsize=14)
#
#ax.tick_params(axis='both', which='major', labelsize=12)
#ax.legend(loc='upper left',fontsize=12)
#plt.show()
