from ngsolve import *
import netgen.gui

from netgen.geom2d import SplineGeometry
import numpy as np
##Setup Geometry
def MakeGeometry(hmax):
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (1, 1), bcs = ("wall", "wall", "wall", "wall"))
    mesh = Mesh( geo.GenerateMesh(maxh=hmax))
    Draw(mesh)
    return mesh

def calcNorm(gfu,sol,mesh):

    #Calc error
    delta_u = sol.components[0]-gfu.components[0]
    delta_du = grad(sol.components[0])-grad(gfu.components[0])
    delta_p = sol.components[2]-gfu.components[2]
    ##Calculate norms
    p_norm = Integrate(delta_p**2,mesh)
    u_norm = Integrate(InnerProduct(delta_u,delta_u),mesh)
    du_norm = Integrate(InnerProduct(delta_du,delta_du),mesh)

    return  [np.sqrt(u_norm), np.sqrt(du_norm), np.sqrt(p_norm)]

def tang(arr,n):
    return arr-InnerProduct(arr,n)*n

def cust_div(arr):
    return CoefficientFunction((arr[0,0]+arr[1,1]))

def cust_grad(arr):
    return CoefficientFunction((arr[0].Diff(x),arr[0].Diff(y),arr[1].Diff(x),arr[1].Diff(y)),dims=(2,2))

def SolveSystem(X,psi,pressure,nu,mesh,alpha,k,enable_HDiv=False):

    #Test and Trial functions
    (u,uhat,p,lam),(v,vhat,q,mu) = X.TnT()

    h = specialcf.mesh_size
    n = specialcf.normal(2)

    #Strain tensor
    #eps_u = 0.5*(cust_grad(u)+cust_grad(u).trans)
    #eps_v = 0.5*(cust_grad(v)+cust_grad(v).trans)
    eps_u = 0.5*(grad(u)+grad(u).trans)
    eps_v = 0.5*(grad(v)+grad(v).trans)

    #Initializes solution and rhs
    u_xe = psi.Diff(y)
    u_ye = -psi.Diff(x)
    sol_e = GridFunction(X)
    sol_e.components[0].Set((u_xe,u_ye))
    sol_e.components[2].Set(pressure)

    grad_ue = cust_grad((u_xe,u_ye))
    eps_nu_ue = -nu*0.5*(grad_ue + grad_ue.trans)
    f = CoefficientFunction((eps_nu_ue[0,0].Diff(x)+eps_nu_ue[0,1].Diff(y),eps_nu_ue[1,0].Diff(x)+eps_nu_ue[1,1].Diff(y)))
    f+= CoefficientFunction((pressure.Diff(x),pressure.Diff(y)))
    #Draw(f, mesh, "f")

    #Assemble BLF
    a = BilinearForm(X)
    if enable_HDiv:
        a +=  (nu*InnerProduct(eps_u,eps_v)-div(u)*q-div(v)*p+lam*q+mu*p)*dx
        a += -nu*InnerProduct(eps_u*n,tang(v-vhat,n))*dx(element_boundary=True)
        a += -nu*InnerProduct(eps_v*n,tang(u-uhat,n))*dx(element_boundary=True)
        a +=   nu*alpha/h*k**2*InnerProduct(tang(u-uhat,n),tang(v-vhat,n))*dx(element_boundary=True)
    else:
        a +=  (nu*InnerProduct(eps_u,eps_v)-cust_div(eps_u)*q-cust_div(eps_v)*p+lam*q+mu*p)*dx
        a += -nu*InnerProduct(eps_u*n,v-vhat)*dx(element_boundary=True)
        a += -nu*InnerProduct(eps_v*n,u-uhat)*dx(element_boundary=True)
        a +=  nu*alpha*k**2*InnerProduct(u-uhat,v-vhat)/h*dx(element_boundary=True)
        a +=  InnerProduct(u-uhat,n)*q*dx(element_boundary=True)
        a +=  InnerProduct(v-vhat,n)*p*dx(element_boundary=True)
    a.Assemble()

    #Assemble LF
    rhs = LinearForm(X)
    rhs += f*v*dx(bonus_intorder=5)
    rhs.Assemble()

    ##Solve System
    gfu = GridFunction(X)
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv*rhs.vec

    return gfu, sol_e

#Set polynomial order, psi, pressure etc.
k = 3
alpha =  1e-2

#Solution and viscosity
psi = (x*(x-1)*y*(y-1))**2
p = x**5+y**5-1/3
#p = x+y-1

offset = 10
nu = np.power(10,(np.arange(-8,offset-8,1,dtype=float)))

#Error
errorHDG = np.zeros((offset,3))
errorHDGDiv = np.zeros((offset,3))

#Generate mesh
mesh = MakeGeometry(0.1)

#HDG Space
V1 = L2(mesh, order=k)
V2 = FacetFESpace(mesh,order=k,dirichlet="wall")
Q = L2(mesh,order=k-1)
N = NumberSpace(mesh)
X = V1**2 * V2**2 * Q * N

for i in range(offset):
    gfu, exact = SolveSystem(X,psi,p,nu[i],mesh,alpha,k)
    errorHDG[i,:] = calcNorm(gfu,exact,mesh)

Draw(gfu.components[0],mesh,"gfu_HDG")
Draw(exact.components[0],mesh,"exact")

#HDG Div Space
V1 = HDiv(mesh, order=k, dirichlet="wall")
V2 = TangentialFacetFESpace(mesh,order=k,dirichlet="wall")
Q = L2(mesh,order=k-1)
N = NumberSpace(mesh)
X = V1 * V2 * Q * N

for i in range(offset):
    gfu, exact = SolveSystem(X,psi,p,nu[i],mesh,alpha,k,True)
    errorHDGDiv[i,:] = calcNorm(gfu,exact,mesh)


Draw(gfu.components[0],mesh,"gfu_HDG_div")

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("GTK3Agg")

plt.figure()
plt.loglog(nu,errorHDG[:,0],marker='o',label=r"$HDG$")
plt.loglog(nu,errorHDGDiv[:,0],marker='o',label=r"$HDG_{Div}$")
plt.grid(True, which="both",linewidth=0.5)
plt.xlabel(r"$\nu$",fontsize=12)
plt.ylabel(r"$|| u - u_h ||_{L^2}$",fontsize=12)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=12)

plt.figure()
plt.loglog(nu,errorHDG[:,1],marker='o',label=r"$HDG$")
plt.loglog(nu,errorHDGDiv[:,1],marker='o',label=r"$HDG_{Div}$")
plt.grid(True, which="both",linewidth=0.5)
plt.xlabel(r"$\nu$",fontsize=12)
plt.ylabel(r"$\| u - u_h \|_{H^1}$",fontsize=12)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=12)

plt.figure()
plt.loglog(nu,errorHDG[:,2],marker='o',label=r"$HDG$")
plt.loglog(nu,errorHDGDiv[:,2],marker='o',label=r"$HDG_{Div}$")
plt.grid(True, which="both",linewidth=0.5)
plt.xlabel(r"$\nu$",fontsize=12)
plt.ylabel(r"$|| p - p_h ||_{L^2}$",fontsize=12)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()