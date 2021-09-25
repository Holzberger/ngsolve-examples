from ngsolve import *
import netgen.gui

from netgen.geom2d import SplineGeometry
import numpy as np

import ngsolve.meshes as ngm
import matplotlib.pyplot as plt
import time

def MakeGeometry(L, hmax=0.1):
    if L>=0: # use refinement
        mesh = ngm.MakeStructured2DMesh(quads=False, nx=2, ny=2)
        for i in range(L+1):
            mesh.Refine()
    else:
        geo  = SplineGeometry()
        geo.AddRectangle( (0, 0), (1, 1) )
        mesh = Mesh( geo.GenerateMesh(maxh=hmax))
    return mesh

def solve_system(L=-1,hmax=0.1, eps=0.01, b_wind=(2,1), k=1, plots=False):
    mesh = MakeGeometry(L=L,hmax=hmax)
    b    = CoefficientFunction(b_wind)
    
    def my_exp(b,x_coord,eps):
        return (exp(b*x_coord/eps)-1)/(exp(b/eps)-1)

    f      = b[0]*( y - my_exp(b[1],y,eps)) + b[1]*( x - my_exp(b[0],x,eps))
    u_exct = ( y - my_exp(b[1],y,eps)) * ( x - my_exp(b[0],x,eps))


    n = specialcf.normal(2)
    h = specialcf.mesh_size
    alpha = 3
    kappa = 1
    #HDG Space
    V     = L2(mesh, order=k)
    V_hat = FacetFESpace(mesh,order=k,dirichlet=[1,2,3,4])
    X     = V * V_hat  
 
    (u,u_hat),(v,v_hat) = X.TnT()
    
    a  = BilinearForm(X)
    # a_HGD, lagrange multiplier
    a += (eps*InnerProduct(grad(u),grad(v)) )*dx
    a += (-eps*InnerProduct(grad(u), n)*(v-v_hat))*dx(element_boundary=True)
    a += (-eps*InnerProduct(grad(v), n)*(u-u_hat))*dx(element_boundary=True)
    a += ((eps*alpha*k*k/h)*(v-v_hat)*(u-u_hat))*dx(element_boundary=True)
    # c_HDG
    u_up = IfPos(b*n, u, u_hat)
    c_Tout = IfPos(b*n, b*n*(u_hat-u)*v_hat, 0)
    a += ( -InnerProduct(u*b, grad(v)) )*dx
    a += ( InnerProduct(b,n)*u_up*v )*dx(element_boundary=True)
    a += ( c_Tout )*dx(element_boundary=True)
    a.Assemble()

    rhs  = LinearForm(X)
    rhs += (f*v)*dx
    rhs.Assemble()

    gfu          = GridFunction(X)
    inv          = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec
    
    if plots:
        Draw(gfu.components[0],mesh,"gfu")
        Draw(f, mesh,"f")
        Draw(u_exct, mesh,"u_exct")

    return u_exct, gfu, mesh

solve_system(plots=True, k=3, eps=0.01,L=3)


