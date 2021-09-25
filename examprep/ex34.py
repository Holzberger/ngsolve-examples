import numpy as np
import matplotlib.pyplot as plt
from ngsolve import *
from ngsolve.solvers import *
import ngsolve.meshes as ngm
from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square


def plot_BD():
    x1 = 0.5
    x_top = np.linspace(0,1,100)
    def u1(x_top, x1):
        ux = x_top*0+1
        ux[x_top<=x1] = 1 - 1/4 *(1-np.cos( (x1-x_top[x_top<=x1])/x1*np.pi ))**2
        ux[x_top>=1-x1] = 1 - 1/4 *(1-np.cos((x_top[x_top>=1-x1]-(1-x1))/x1*np.pi))**2
        return ux 

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(x_top, u1(x_top, 0.2),label="0.2")
    ax.plot(x_top, u1(x_top, 0.5),label=0.5)
    ax.plot(x_top, u1(x_top, 0.9),label=0.9)
    ax.legend(loc="best")
    plt.savefig("cavity_BDvel.pdf")


def mesh_rectangle(hmax=0.1):
    #geo = SplineGeometry()
    #geo.AddRectangle( (0, 0), (1, 1))
    #mesh = Mesh( geo.GenerateMesh(maxh=hmax))
    mesh = Mesh(unit_square.GenerateMesh(maxh=hmax))
    return mesh

def mesh_byrefinement(L=0):
    mesh = ngm.MakeStructured2DMesh(quads=False,nx=2,ny=2)
    for i in range(L+1):
        mesh.Refine()
    return mesh

def get_P2bubble(meshi):
    V = VectorH1(mesh, order=2, dirichlet = "bottom|right|left|top")
    V.SetOrder(TRIG, 3)
    V.Update()
    Q = L2(mesh, order=1)
    return [V,Q]




def Solve_cavity(VxQ, nu=1, x1=0.5, newton=True, show_plots=False):

    X = FESpace(VxQ)

    (u,p),(v,q) = X.TnT()

    def eps(u): 
        return 1/2*(grad(u) + grad(u).trans)
    def a(u,v):
        return nu*InnerProduct(grad(u), grad(v))
    def b(u,q):
        return -(grad(u)[0,0]+grad(u)[1,1])*q
    def c(w,u,v):
        return InnerProduct(grad(u)*w,v)

    # solving stokes
    A = BilinearForm(X)
    A += a(u,v)*dx
    A += ( b(v,p) + b(u,q) - 1e-10*p*q)*dx
    A.Assemble()
  
    L = LinearForm(X)
    L.Assemble()

    gfu = GridFunction(X)
   
    # add u with BC but f=0
    #uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
    uin  = CoefficientFunction(1)
    uin1 = CoefficientFunction(1-1/4*(1-cos( (x1-x)/x1*np.pi ))**2)
    uin2 = CoefficientFunction(1-1/4*(1-cos( (x-(1-x1))/x1*np.pi ))**2)
    uin  = IfPos(x-x1, uin, uin1)
    uin  = IfPos(x-(1-x1), uin2, uin)
    utop = CoefficientFunction((uin,0))
    if show_plots:
        Draw(utop, mesh, "utop")
    
    gfu.components[0].Set(utop, definedon=mesh.Boundaries("top"))
    res = L.vec.CreateVector()
    # homogenize solution
    res.data      = - A.mat * gfu.vec # get residual f-Ax
    inv_stokes    = A.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data += inv_stokes * res

    # residuals for nonlinear iteration
    def ruh(uk,pk,v):
        return - c(uk,uk,v) - a(uk,v) - b(v,pk)
    def rph(uk,q):
        return -b(uk,q) 


    
    A1 = BilinearForm(X)
    # newton
    if newton:
        A1 += SymbolicBFI(c(gfu.components[0], u, v) + c(u, gfu.components[0], v)\
                          +a(u,v) + b(v, p) + b(u,q) - 1e-10*p*q) 
    else: # picard
        A1 += SymbolicBFI(c(gfu.components[0], u, v) + a(u,v) + b(v, p) + b(u,q) - 1e-10*p*q ) 
    
    
    l_ruh = BilinearForm(X)
    l_ruh += ruh(u, p, v)*dx
    

    l_rph = BilinearForm(X)
    l_rph += rph(u,q)*dx
    
    
    L1 = BilinearForm(X)
    L1 += ruh(u, p, v)*dx + rph(u,q)*dx + (1e-10*p*q)*dx
  
    
    l_ruh.Assemble()
    l_rph.Assemble()
    L1.Assemble()


    du = gfu.vec.CreateVector()
    gfu_old = gfu.vec.CreateVector()

    r_uh = gfu.vec.CreateVector()
    r_ph = gfu.vec.CreateVector()
    l1   = gfu.vec.CreateVector()
    DU   = GridFunction(X)
    counter=0
    for it in range(300): 
        print("iteration ",it)
        A1.Assemble()
        L1.Apply(gfu.vec, l1)
        next_inv = A1.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
        DU.vec.data  = next_inv * l1

        gfu.vec.data += DU.vec.data
        print(InnerProduct(DU.vec,DU.vec))
       
        l_ruh.Apply(gfu.vec, r_uh)
        l_rph.Apply(gfu.vec, r_ph)
       
        error= sqrt(abs(InnerProduct(DU.vec,l1 )))
        print(error)
        counter +=1
        if error<1e-10:
            break
        elif error>1e10:
            counter = -1
            break
        elif it==299:
            counter = -2

    if show_plots:
        Draw(gfu.components[0], mesh, "vel")
        Draw(gfu.components[1], mesh, "pressure")
        Draw(Norm(gfu.components[0]), mesh, "|vel|")
        SetVisualization(max=2)
    return gfu, counter


#iter_counts = []
#
#L = [0,1,2,3]
#Nu = [0.01, 0.002, 0.001]
#
#
#for nu in Nu:
#    for l in L:
#        mesh = mesh_byrefinement(L=l)
#        VxQ  = get_P2bubble(mesh)
#        gfu, counter  = Solve_cavity(VxQ, nu=nu, x1=0.1,newton=True)
#        iter_counts.append(counter)
#        mesh = mesh_byrefinement(L=l)
#        VxQ  = get_P2bubble(mesh)
#        gfu, counter  = Solve_cavity(VxQ, nu=nu, x1=0.1,newton=False)
#        iter_counts.append(counter)
#print(iter_counts)


mesh = mesh_byrefinement(L=3)
VxQ  = get_P2bubble(mesh)
gfu, counter  = Solve_cavity(VxQ, nu=0.001, x1=0.1,newton=False,show_plots=True)
