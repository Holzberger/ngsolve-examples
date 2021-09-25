from ngsolve import *
import netgen.gui

from netgen.geom2d import SplineGeometry
import numpy as np

def MakeGeometry(hmax):
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (1, 1), bcs = ("wall", "wall", "wall", "wall"))
    mesh = Mesh( geo.GenerateMesh(maxh=hmax))
    return mesh



def SolveSystem(X,psi,pres,nu,mesh,alpha,k,enable_HDiv=False):

    (u,u_hat,p,lam),(v,v_hat,q,mu) = X.TnT()

    h = specialcf.mesh_size
    n = specialcf.normal(2)

    # construct exact solution
    u_ex = CoefficientFunction( (psi.Diff(y), -psi.Diff(x)) )
    sol_ex = GridFunction(X)
    sol_ex.components[0].Set(u_ex)
    sol_ex.components[2].Set(pres)

    # generate a RHS
    grad_ue = CoefficientFunction( (u_ex[0].Diff(x),
                           u_ex[0].Diff(y),
                           u_ex[1].Diff(x),
                           u_ex[1].Diff(y)),
                           dims=(2,2))
    eps_nu_uex = -1/2*nu*(grad_ue + grad_ue.trans)
    f = CoefficientFunction((   eps_nu_uex[0,0].Diff(x)+eps_nu_uex[0,1].Diff(y),
                       eps_nu_uex[1,0].Diff(x)+eps_nu_uex[1,1].Diff(y)))
    f += CoefficientFunction((pres.Diff(x), pres.Diff(y)))

    # generate HDG BLF
    def eps(u):
        return 1/2*(grad(u)+grad(u).trans)
    def my_div(u):
        return grad(u)[0,0]+grad(u)[1,1]
    a  = BilinearForm(X)
    a += (nu*InnerProduct(eps(u),eps(v)) + lam*q + mu*p )*dx
    a += (-nu*InnerProduct(eps(u)*n, v-v_hat))*dx(element_boundary=True)
    a += (-nu*InnerProduct(eps(v)*n, u-u_hat))*dx(element_boundary=True)
    a += ((nu*alpha*k*k/h)*InnerProduct(v-v_hat, u-u_hat))*dx(element_boundary=True)
    a += (-my_div(u)*q)*dx + (InnerProduct(u-u_hat,n*q))*dx(element_boundary=True)

    a += (-my_div(v)*p)*dx + (InnerProduct(v-v_hat,n*p))*dx(element_boundary=True)
    a.Assemble()

    rhs  = LinearForm(X)
    rhs += f*v*dx(bonus_intorder=5)
    rhs.Assemble()

    #Solve System
    gfu = GridFunction(X)
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec


    return gfu, sol_ex

nu = 100
h = 0.5
k = 3
alpha = 1e-2

psi = (x*(x-1)*y*(y-1))**2
pres = x**5+y**5-1/3


mesh = MakeGeometry(h)

#HDG Space
V = L2(mesh, order=k)
V_hat = FacetFESpace(mesh,order=k,dirichlet="wall")
Q = L2(mesh,order=k-1)
N = NumberSpace(mesh)
X = V**2 * V_hat**2 * Q * N




gfu, exact = SolveSystem(X, psi, pres, nu, mesh, alpha, k)

Draw(gfu.components[0],mesh,"gfu_HDG")
Draw(exact.components[0],mesh,"exact")




