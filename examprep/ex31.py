from ngsolve import *
import netgen.gui

from netgen.geom2d import SplineGeometry
import numpy as np

import ngsolve.meshes as ngm

import matplotlib.pyplot as plt
import time

def MakeGeometry(L):
    mesh = ngm.MakeStructured2DMesh(quads=False, nx=2, ny=2)
    for i in range(L+1):
        mesh.Refine()
    return mesh

def solve_system(L=1, eps=0.01, b_wind=(2,1), k=1, plots=False):
    mesh = MakeGeometry(L)
    b    = CoefficientFunction(b_wind)
    
    def my_exp(b,x_coord,eps):
        return (exp(b*x_coord/eps)-1)/(exp(b/eps)-1)

    f      = b[0]*( y - my_exp(b[1],y,eps)) + b[1]*( x - my_exp(b[0],x,eps))
    u_exct = ( y - my_exp(b[1],y,eps)) * ( x - my_exp(b[0],x,eps))

    V    = H1(mesh, order=k, dirichlet="bottom|right|left|top")
    u, v = V.TnT()

    a  = BilinearForm(V)
    a += (eps*InnerProduct(grad(u), grad(v)))*dx
    a += (InnerProduct(b, grad(u))*v)*dx
    a.Assemble()

    rhs  = LinearForm(V)
    rhs += (f*v)*dx
    rhs.Assemble()

    gfu          = GridFunction(V)
    try:
        inv          = a.mat.Inverse(freedofs=V.FreeDofs(), inverse="pardiso")
    except:
        inv          = a.mat.Inverse(freedofs=V.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec
    
    if plots:
        Draw(gfu,mesh,"gfu")
        Draw(f, mesh,"f")
        Draw(u_exct, mesh,"u_exct")

    return u_exct, gfu, mesh

def solve_SUPG(L=1, eps=0.01, b_wind=(2,1), k=1, 
               plots=False, stabilize=True):

    mesh = MakeGeometry(L=L)
    b    = CoefficientFunction(b_wind)
    
    def my_exp(b,x_coord,eps):
        return (exp(b*x_coord/eps)-1)/(exp(b/eps)-1)

    f      = b[0]*( y - my_exp(b[1],y,eps)) + b[1]*( x - my_exp(b[0],x,eps))
    u_exct = ( y - my_exp(b[1],y,eps)) * ( x - my_exp(b[0],x,eps))

    #b_abs = InnerProduct(b,b)**0.5
    b_abs = (b[0]**2+b[1]**2)**0.5 
    # get the mesh peclet number
    h = specialcf.mesh_size
    Ph = b_abs*h/eps
    # calc stabilisation parameter
    #alpha = IfPos(Ph-1, h/b_abs, 0)
    alpha = h/b_abs

    V    = H1(mesh, order=k, dirichlet="bottom|right|left|top")
    u, v = V.TnT()

    a  = BilinearForm(V)
    a += (eps*InnerProduct(grad(u), grad(v)))*dx
    a += (InnerProduct(b, grad(u))*v)*dx
    if stabilize:
        u_H = u.Operator("hesse")
        Del_u = u_H[0,0] + u_H[1,1]
        rh = -eps*Del_u + InnerProduct(b,grad(u))
        a += (alpha*rh*InnerProduct(b, grad(v)))*dx
    a.Assemble()

    rhs  = LinearForm(V)
    rhs += (f*v)*dx
    if stabilize:
        rhs +=  (alpha*f*InnerProduct(b,grad(v)))*dx
    rhs.Assemble()

    gfu          = GridFunction(V)
    try:
        inv          = a.mat.Inverse(freedofs=V.FreeDofs(), inverse="pardiso")
    except:
        inv          = a.mat.Inverse(freedofs=V.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec
    
    
    if plots:
        Draw(gfu,mesh,"gfu")
        Draw(f, mesh,"f")
        Draw(u_exct, mesh,"u_exct")
        Draw(Ph, mesh,"Ph")
        Draw(alpha, mesh,"alpha")

    return u_exct, gfu, mesh

def solve_HDG(L=1, eps=0.01, b_wind=(2,1), k=1, plots=False):
    mesh = MakeGeometry(L=L)
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
    V_hat = FacetFESpace(mesh,order=k,dirichlet="bottom|right|left|top")
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
    try:
        inv          = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="pardiso")
    except:
        inv          = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec
    
    if plots:
        Draw(gfu.components[0],mesh,"gfu")
        Draw(f, mesh,"f")
        Draw(u_exct, mesh,"u_exct")

    gfu_out          = GridFunction(V)
    gfu_out.vec.data = gfu.components[0].vec.data
    return u_exct, gfu_out, mesh

def get_err_norm(gfu, u_exct, mesh):
    delta_u = u_exct-gfu
    u_norm  = Integrate(InnerProduct(delta_u,delta_u),mesh)
    u_L2  = u_norm**0.5
    return u_L2

def run_ex(ex_nr=1):
    # refinements
    L = [0,1,2,3,4]
    # orders
    K = [1,2,3]
    # L2 error
    errs = []

    for k in K:
        errs.append([])
        for l in L:
            if(ex_nr==1):
                u_exct, gfu , mesh = solve_system(L=l,k=k)
            elif(ex_nr==2):
                u_exct, gfu , mesh = solve_SUPG(L=l, k=k)
            elif(ex_nr==3):
                u_exct, gfu , mesh = solve_HDG(L=l, k=k)
            u_L2 = get_err_norm(gfu, u_exct, mesh)
            errs[-1].append(u_L2)
            print(u_L2)
    plt.ioff()
    fig, ax = plt.subplots()
    for err, k in zip(errs,K):
        ax.semilogy( L, err,".-",label="order= {}".format(k))
    for n in range(1,4):
        ax.semilogy(L,(1/2)**(n*np.array(L)),"--",color="black")
    ax.legend(loc='best',fontsize=12)
    ax.grid(True, which="both",linewidth=0.5)
    ax.set_xlabel('L level of refinement',fontsize=14)
    ax.set_ylabel("error",fontsize=14)
    ax.set_ylim([1e-4,1e-0])
    plt.show()
    plt.savefig("norms_ex3{}.pdf".format(ex_nr))

def plot_line(px, py=0.5):
    plt.ioff()
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(3,4,hspace=0, wspace=0)
    ax = gs.subplots(sharex="col", sharey="row")
    
     # refinements
    L = [0,1,2,3]
    # orders
    K = [1,2,3]
    for k in K:
        for l in L:
            u_exct, gfu , mesh = solve_system(L=l,k=k,)
            vals = [gfu(mesh(p,py)) for p in px]
            ax[k-1,l].plot(px,vals)
            u_exct, gfu , mesh = solve_SUPG(L=l, k=k)
            vals = [gfu(mesh(p,py)) for p in px]
            ax[k-1,l].plot(px,vals)
            u_exct, gfu , mesh = solve_HDG(L=l, k=k)
            vals = [gfu(mesh(p,py)) for p in px]
            ax[k-1,l].plot(px,vals)
            #ax[k-1,l].set_ylim([-0.6,0.6])
    fig.text(0.5,0.03,"x at y=0.5, left L=0 right L=3",ha="center")
    fig.text(0.03,0.5,"fuction value, top k=1 bottom k=3",va="center", rotation="vertical")
    plt.show()
    #plt.savefig("cut_ex3_123.pdf")




#solve_system(L=3, eps=0.01, k=3, plots=True)
#run_ex(ex_nr=1)
run_ex(ex_nr=2)
#run_ex(ex_nr=3)
#plot_line(np.linspace(0,1,100),py=0.5)
#run_ex32()
#u_exct, gfu , mesh = solve_SUPG(hmax=0.08, k=3,plots=True,
#                                eps=0.01, stabilize=True)





