from ngsolve import *
import netgen.gui

from netgen.geom2d import unit_square
from netgen.geom2d import SplineGeometry
#import matplotlib.pyplot as plt
import numpy as np
#import scipy.sparse as sp

nu = 1e-5

phi = x**2*(1-x)**2*y**2*(1-y)**2

u_sol = CoefficientFunction((-phi.Diff(y), phi.Diff(x)))
p_sol = x+y-1#x**5+y**5-1/3
# setup strain rate tensor
du = CoefficientFunction( ((u_sol[0].Diff(x), u_sol[0].Diff(y)),
                           (u_sol[1].Diff(x), u_sol[1].Diff(y)) ),dims=(2,2) )
eps_u = 1/2*(du + du.trans)

f =  CoefficientFunction( ( (-2 * nu *(eps_u[0,0].Diff(x) + eps_u[0,1].Diff(y)) + p_sol.Diff(x)),
                            (-2 * nu *(eps_u[1,0].Diff(x) + eps_u[1,1].Diff(y)) + p_sol.Diff(y))) )

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
Draw(mesh)

#P2+c
V = VectorH1(mesh, order=2, dirichlet=".*")
V.SetOrder(TRIG,3)
# P1c
Q = L2(mesh, order=1)
N = NumberSpace(mesh)
X = FESpace([V,Q,N])

(u,p,lam),(v,q,mu) = X.TnT()

eps_u = 1/2*(grad(u)+grad(u).trans)
eps_v = 1/2*(grad(v)+grad(v).trans)

a = BilinearForm(X)
a += (nu*InnerProduct(eps_u,eps_v)-div(v)*p-div(u)*q)*dx
a += (p*mu+q*lam)*dx
a.Assemble()

f_lf = LinearForm(X)
#f_lf += f*v.Operator("divfree_reconstruction")*dx
f_lf += f*v*dx
f_lf.Assemble()

gfu = GridFunction(X)
inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
gfu.vec.data = inv*f_lf.vec

Draw(gfu.components[0], mesh, "fem_vel")
Draw(gfu.components[1], mesh, "fem_pres")

res = f_lf.vec.CreateVector()
res.data =f_lf.vec- a.mat*gfu.vec

force = GridFunction(V)
force.Set(f)
Draw(force.components[1], mesh, "force")

exact_sol = GridFunction(X)
exact_sol.components[0].Set((u_sol[0],u_sol[1]))
exact_sol.components[1].Set(p_sol)


delta_du = grad(exact_sol.components[0])-grad(gfu.components[0])
# H1 seminorm
du_norm = Integrate(InnerProduct(delta_du,delta_du),mesh)

print(du_norm)
#Draw(res.vec, mesh, "res")

#err = Integrate(res.data, mesh, VOL, element_wise=True)