from ngsolve import *
from netgen.geom2d import SplineGeometry
import matplotlib.pyplot as plt;
import numpy as np
import time
ngsglobals.msg_level = 0

def solve_system(tau, tend, el_order, theta,newton):
    # viscosity
    nu = 0.001
    # timestepping parameters
    #tau = 0.04
    # simulation time
    #tend = 5
    # element order
    #el_order = 3
    # use newton or picard iteration
    #newton = True
    # theta value for phi scheme
    #theta  = 0.5
    #%% create domain with cylinder
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
    geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.009)
    mesh = Mesh( geo.GenerateMesh(maxh=0.05) )
    mesh.Curve(el_order+1)
    #%% define FE spaces
    V = VectorH1(mesh,order=el_order, dirichlet="wall|cyl|inlet")
    #V.SetOrder(TRIG, 3)
    #V.Update()
    Q = H1(mesh,order=el_order-1)
    X = FESpace([V,Q])
    u,p = X.TrialFunction()
    v,q = X.TestFunction()
    #%% solve stokes eq to init solution
    stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-10*p*q
    a      = BilinearForm(X)
    a     += stokes*dx
    a.Assemble()
    # nothing here ...
    f = LinearForm(X)   
    f.Assemble()
    # gridfunction for the solution
    gfu = GridFunction(X)
    # parabolic inflow at inlet:
    uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
    # solve Stokes problem for initial conditions:
    inv_stokes = a.mat.Inverse(X.FreeDofs())
    print("number of free DOFS ",np.sum(np.array(X.FreeDofs(), dtype="int")))
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat*gfu.vec
    gfu.vec.data += inv_stokes * res
    # plot velocity-mag
    Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)
    #%% setup functionals for NS equation
    # NS functionals
    def a(u, v):
        return nu*InnerProduct(grad(u), grad(v))
    def b(u,q):
        return -(grad(u)[0,0]+grad(u)[1,1])*q
    def c(w,u,v):
        return InnerProduct(grad(u)*w,v)
    # save solution from latest converged time step
    gfu0 = GridFunction(X)
    gfu0.vec.data = gfu.vec.data
    u0 = gfu0.components[0]
    p0 = gfu0.components[1]
    # define BLF for iterative scheme
    A = BilinearForm(X)
    A += SymbolicBFI(u*v + tau*theta*( a(u,v) + c(gfu.components[0], u, v) )\
                         + tau*( b(v,p) + b(u,q)  - 1e-10*p*q ) ) 
    if newton: # add extra term for newton method
        A += SymbolicBFI(tau*theta*c(u, gfu.components[0], v) ) 
    # define LF as BLF where u will be given
    L = BilinearForm(X)
    L += SymbolicBFI( u0*v - u*v\
                        + tau*( (1-theta)*( -a(u0,v)-c(u0, u0,v) )\
                        + theta*(-a(u,v)-c(u,u,v))\
                        - b(v,p) - b(u,q) +1e-10*p*q) )
    #%% setup integrals for drag and lift coeff. calculation
    # get domains normal vectors (correct direction)
    n = -specialcf.normal(2)
    #n = CoefficientFunction((-n[0],-n[1]))
    # get domains tangential vectors
    tang = CoefficientFunction((n[1],-n[0]))
    # define integrads of drag and lift coeff.
    bfv_cd = BoundaryFromVolumeCF(20*( nu*InnerProduct(grad(gfu.components[0])*n,tang)*n[1]-gfu.components[1]*n[0]) )
    bfv_cl = BoundaryFromVolumeCF(-20*( nu*InnerProduct(grad(gfu.components[0])*n,tang)*n[0]+gfu.components[1]*n[1]) )
    #%% solve NS equation iterative
    # init sim.time
    t = 0
    # save time, drag and lift 
    time_vals = []
    drag      = []
    lift      = []
    p1        = []
    p2        = []
    # save increments of nonlinear solution here
    du = GridFunction(X)
    # save LHS. of newton/picard iteration here
    l  = gfu.vec.CreateVector()
    sim_time = time.time()
    # start solver-loop
    with TaskManager(): # enable multi threading 
        while(t<=tend): # time step loop 
            print("=====time {:0.6f}=====".format(t))
            for it in range(20): # nonlinear-solver iteration
                A.Assemble()
                L.Apply(gfu.vec, l)
                next_inv = A.mat.Inverse(freedofs=X.FreeDofs(), inverse="pardiso")
                du.vec.data  = next_inv * l
                gfu.vec.data += du.vec.data
                inc_next = sqrt(InnerProduct(du.vec,du.vec))
                err_next = sqrt(InnerProduct(l,du.vec))
                print ("\rerr={:0.4E}, increment {:0.4E}".format(err_next, inc_next), end="")
                if err_next<1e-10: # check tolerance for iteration
                    break
            Redraw()
            # save converged timestep
            gfu0.vec.data = gfu.vec.data
            u0 = gfu.components[0]
            p0 = gfu.components[1] 
            # increment time
            t += tau
            time_vals.append(t)
            # save drag and lift
            lift.append(Integrate(bfv_cl, mesh, definedon=mesh.Boundaries("cyl")))
            drag.append(Integrate(bfv_cd, mesh, definedon=mesh.Boundaries("cyl")))
            print("\nlift {:0.4f}".format(lift[-1]), "maxminlift {:0.4f}, {:0.4f}".format(np.max(lift), 
                                                                                 np.min(lift)))
            print("drag {:0.4f}".format(drag[-1]), "maxdrag {:0.4f}".format(np.max(drag)))
            # save p1 p2 pressures
            p1.append(gfu.components[1](mesh(0.15,0.2)))
            p2.append(gfu.components[1](mesh(0.25,0.2)))
    sim_time = time.time()-sim_time
    time_vals = np.array(time_vals)
    drag      = np.array(drag)
    lift      = np.array(lift)
    p1        = np.array(p1)
    p2        = np.array(p2)
    path = "./ex35data/"
    filename = path+"solution_{}_{}_{}_{}_{:0.2f}_".format(el_order,tau,newton,theta,sim_time)
    np.save(filename, np.array([time_vals,drag,lift,p1,p2]))



#solve_system(tau,tend,el_order,theta,newton=True):
#solve_system(0.001, 1, 2, 0,True)
#solve_system(0.01, 5, 3, 0.5,True)
#solve_system(0.01, 5, 3, 0.75,True)
#solve_system(0.01, 5, 3, 1,True)
#solve_system(0.01, 5, 4, 0.5,True)

solve_system(0.0005, 2, 3, 0,True)
solve_system(0.001, 2, 3, 0.25,True)
solve_system(0.01, 2, 3, 0.5,True)
solve_system(0.01, 2, 3, 0.75,True)
solve_system(0.01, 2, 3, 1,True)

solve_system(0.01, 5, 5, 0.5,True)
solve_system(0.005, 5, 5, 0.5,True)
