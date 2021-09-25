from ngsolve import *
import netgen.gui

from netgen.geom2d import unit_square
from netgen.geom2d import SplineGeometry
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp





k = 2
alpha = 4#1e3


mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
h    = specialcf.mesh_size


V = H1(mesh, order=k)


u = V.TrialFunction()
v = V.TestFunction()

a  = BilinearForm(V)

a += grad(u)*grad(v)*dx 
a += -grad(u)*specialcf.normal(2)*v*ds(skeleton=True)
a += -grad(v)*specialcf.normal(2)*u*ds(skeleton=True)
a += alpha*k**2/h*u*v*ds(skeleton=True)

c = Preconditioner(a, "local")



f = LinearForm(V)

uD =  1
f += 30*cos(x)*cos(y) *v*dx 
f += -grad(v)*specialcf.normal(2)*uD*ds(skeleton=True)
f += alpha*k**2/h*uD*v*ds(skeleton=True)



gfu = GridFunction(V)

V.Update()
a.Assemble()
f.Assemble()
gfu.Update()
## Conjugate gradient solver
inv = CGSolver(a.mat, c.mat, printrates = True, precision = 1e-8, maxsteps = 10000)
gfu.vec.data = inv * f.vec



#gfu.vec.data = a.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky") * f.vec

Draw (gfu)

rows,cols,vals = a.mat.COO()
A = sp.csr_matrix((vals,(rows,cols)))
print("condition number: {}".format(np.linalg.cond(A.todense())))


