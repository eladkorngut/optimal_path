import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import least_squares
from scipy.integrate import ode
import scipy.linalg as la
from scipy.integrate import odeint
from pylab import figure, plot, xlabel, grid, legend, title,savefig
from matplotlib.font_manager import FontProperties
import csv


ecl_dist=lambda xi,yi,xf,yf: np.sqrt((xi-xf)**2+(yi-yf)**2)
dx_dt = lambda i,p,b,k: b*(1-i)*i*np.exp(p)-k*i*np.exp(-1*p)
dp_dt = lambda i,p,b,k: b*(2*i-1)*(np.exp(p)-1)-k*(np.exp(-1*p)-1)

def vectorfiled(q,t,m):
    x,p = q
    b,k = m
    f=[dx_dt(x,p,b,k),dp_dt(x,p,b,k)]
    return f


def shoot(x0,p0,dx_dt,dp_dt,dt,pf,xf):
    x_path,p_path = [x0], [p0]
    integrator_x = ode(dx_dt).set_integrator('dopri5')
    integrator_x.set_initial_value(x0,t=0)
    integrator_p = ode(dp_dt).set_integrator('dopri5')
    integrator_p.set_initial_value(p0,t=0)
    while p_path[-1]>pf and x_path[-1]>xf and p_path[-1]<=0:
        integrator_x.integrate(integrator_x.t+dt)
        integrator_p.integrate(integrator_p.t + dt)
        if np.abs(integrator_x.y[0]-x_path[-1])>=0.1 or np.abs(integrator_p.y[0]-p_path[-1])>=0.1:
            x_path.append(integrator_x.y[0])
            p_path.append(integrator_p.y[0])
    resdiual = np.min([ecl_dist(i,j,xf,pf) for i, j in zip(x_path, p_path)])
    return (x_path,p_path),resdiual

def shoot_ode(x0,p0,b,k):
    # ODE parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = 10.0
    numpoints = 250

    # create the time samples for the output of the ODE solver
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    q0 = (x0,p0)
    m0 = (b,k)
    qsol = odeint(vectorfiled,q0,t,args=(m0,),atol=abserr,rtol=relerr)

    f=open('temp_print_numbers.csv','w+')
    with f:
        writer =csv.writer(f)
        writer.writerows(qsol)

    return qsol



# The begining of the original program
paths,residual=[],[]
k,b,r,dt=1.0,5,0.1,0.001
theta=np.linspace(np.pi/100,np.pi,20)
x0,p0,xf,pf =1-k/b,0,0,np.log(k/b)
xi = [x0-r*np.cos(t) for t in theta]
pi = [p0 -r*np.sin(t) for t in theta]
for x,p in zip(xi,pi):
    J = np.array([[(b*np.exp(p)-k*np.exp(-p))-2*b*x*np.exp(0),b*(1-x)*np.exp(p)+k*x*np.exp(-p)],[2*b*np.exp(p),b*(2*x-1)*np.exp(p)+k*np.exp(-p)]])
    eigen_value,eigen_vec=la.eig(J)
    if  eigen_value[0].real>0:
        eig_value, eig_vec = eigen_value[0].real , eigen_vec[:,0].reshape(2,1)
    else:
        eig_value, eig_vec = eigen_value[1].real , eigen_vec[:,1].reshape(2,1)
    # eig_value,direction_vec = J_diag[0].real[0] , J_diag[1][0] if J_diag[0].real[0]<0 else J_diag[0].real[1] , J_diag[1][1]
    # path_current,residual_current=shoot(x0+float(eig_vec[0])*dt,p0+float(eig_vec[1])*dt,dx_dt,dp_dt,dt,pf,xf)
    solution=shoot_ode(x0+float(eig_vec[0])*dt,p0+float(eig_vec[1])*dt,b,k)
    # paths.append(path_current)
    # residual.append(residual_current)
# print('Lowest value',np.min(residual))
# print('choosen path',np.where(residual==np.min(residual)))
# print('this no love song')