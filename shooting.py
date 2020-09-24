import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import least_squares
from scipy.integrate import ode
import scipy.linalg as la
from scipy.integrate import odeint
from pylab import figure, plot, xlabel, grid, legend, title,savefig,ylabel
from matplotlib.font_manager import FontProperties
import csv
import functools
import numdifftools as ndft



def onedshooting():
    ecl_dist = lambda r0, rf: np.sqrt((r0[0] - rf[0]) ** 2 + (r0[1] - rf[1]) ** 2)
    dx_dt = lambda i, p, b, k: b * (1 - i) * i * np.exp(p) - k * i * np.exp(-1 * p)
    dp_dt = lambda i, p, b, k: b * (2 * i - 1) * (np.exp(p) - 1) - k * (np.exp(-1 * p) - 1)
    find_path_distance = lambda qsol, rf: [ecl_dist(q, rf) for q in qsol]

    def vectorfiled(q, t, m):
        x, p = q
        b, k = m
        f = [dx_dt(x, p, b, k), dp_dt(x, p, b, k)]
        return f

    def shoot_ode(x0, p0, t, b, k, abserr, relerr):
        # ODE parameters
        q0 = (x0, p0)
        m0 = (b, k)
        qsol = odeint(vectorfiled, q0, t, args=(m0,), atol=abserr, rtol=relerr)
        return qsol

    # ODE parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = 4.0
    numpoints = 250

    # create the time samples for the output of the ODE solver
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    k,b,r,dt=1.0,5,0.1,0.0001
    paths,residual=[],[]
    theta=np.linspace(np.pi/100,np.pi,20)
    x0,p0,xf,pf =1-k/b,0,0,np.log(k/b)
    xi = [x0-r*np.cos(t) for t in theta]
    pi = [p0 -r*np.sin(t) for t in theta]
    for x,p in zip(xi,pi):
        J = np.array([[(b*np.exp(p)-k*np.exp(-p))-2*b*x*np.exp(p),b*(1-x)*np.exp(p)+k*x*np.exp(-p)],[2*b*np.exp(p),b*(2*x-1)*np.exp(p)+k*np.exp(-p)]])
        eigen_value,eigen_vec=la.eig(J)
        if  eigen_value[0].real>0:
            eig_value, eig_vec = eigen_value[0].real , eigen_vec[:,0].reshape(2,1)
        else:
            eig_value, eig_vec = eigen_value[1].real , eigen_vec[:,1].reshape(2,1)
        current_path=shoot_ode(x0+float(eig_vec[0])*dt,p0+float(eig_vec[1])*dt,t,b,k,abserr,relerr)
        paths.append(current_path)
        path_distances = find_path_distance(current_path, (xf,pf))
        residual.append(min(path_distances))
    index_of_best_path = residual.index(max(residual))
    x_path,  p_path = [a for a in paths[index_of_best_path][0]], [b for b in paths[index_of_best_path][1]]
    figure(1)
    plot(paths[index_of_best_path][:,0],paths[index_of_best_path][:,1])
    xlabel('I')
    ylabel('p')
    title('Optimal path p vs I shooting')
    plt.show()
    savefig('best_path.png',dpi=100)
    print('This no love song')


def hetro_degree_shooting():
    lam, k_avg, epsilon =1.6,50.0,0.1
    ecl_dist = lambda r0, rf: np.sqrt((r0[0] - rf[0]) ** 2 + (r0[1] - rf[1]) ** 2+(r0[2] - rf[2]) ** 2+(r0[3] - rf[3]) ** 2)
    dw_dt = lambda w, u, p_w, p_u: ((1/2)*lam*k_avg*((1/2)*(1-epsilon)*(-u-w+1)*np.exp((p_u+p_w)/2)+(1/2)*(epsilon+1)*(u-w+1)*np.exp((p_w-p_u)/2))
            -(1/4)*(w-u)*np.exp((p_u-p_w)/2)-(1/4)*(u+w)*np.exp((-p_u-p_w)/2))
    du_dt = lambda w, u, p_w, p_u: ((1/2)*lam*k_avg*(w-u*epsilon)*((1/2)*(1-epsilon)*(-u-w+1)*np.exp((p_u+p_w)/2)-(1/2)*(epsilon+1)*(u-w+1)*np.exp((p_w-p_u)/2))
            +(1/4)*(w-u)*np.exp((p_u-p_w)/2)-(1/4)*(u+w)*np.exp((-p_u-p_w)/2))
    dp_w_dt = lambda w, u, p_w, p_u: -((1/2)*lam*k_avg*((1-epsilon)*(-u-w+1)*(np.exp((p_u+p_w)/2)-1)+(1+epsilon)*(u-w+1)*(np.exp((p_u-p_w)/2)-1))
            +(1/2)*lam*k_avg*(w-u*epsilon)*((1-epsilon)*(-(np.exp((p_u+p_w)/2)-1))-(epsilon+1)*(np.exp((p_w+p_u)/2)-1))
            +(1/2)*(np.exp((-p_u-p_w)/2)-1)+(1/2)*(np.exp((p_u-p_w)/2)-1))
    dp_u_dt = lambda w, u, p_w, p_u:-((1/2)*lam*k_avg*(w-epsilon*u)*((epsilon+1)*(np.exp((p_w-p_u)/2)-1)-(1-epsilon)*(np.exp((p_u+p_w)/2)-1))
            -(1/2)*lam*k_avg*epsilon*((1-epsilon)*(-u-w+1)*(np.exp((p_u+p_w)/2)-1)+(epsilon+1)*(u-w+1)*(np.exp((p_w-p_u)/2)-1))
            +(1/2)*(np.exp((-p_u-p_w)/2)-1)+(1/2)*(1-np.exp((p_u-p_w)/2)))
    find_path_distance = lambda qsol, rf: [ecl_dist(q, rf) for q in qsol]
    dq_dt = lambda q:np.array([dw_dt(q[0], q[1], q[2], q[3]),du_dt(q[0], q[1], q[2], q[3]),dp_w_dt(q[0], q[1], q[2], q[3]),dp_u_dt(q[0], q[1], q[2], q[3])])
    H = lambda q: (lam*k_avg/2)*(q[0]-epsilon*q[1])*((1-epsilon)*(1-q[0]-q[1])*(np.exp((q[2]+q[3])/2)-1)+(1+epsilon)*(1-(q[2]-q[3]))*(np.exp((q[2]-q[3])/2)-1))+((q[0]+q[1])/2)*(np.exp(-(q[2]+q[3])/2)-1)+((q[0]-q[1])/2)*(np.exp(-(q[2]-q[3])/2)-1)
    Jacobian_H = ndft.Jacobian(H)
    temp = Jacobian_H([0.37,-0.0234,-0.926,0.075])
    temp2 = [dw_dt(0.37,-0.0234,-0.926,0.075),du_dt(0.37,-0.0234,-0.926,0.075),dp_w_dt(0.37,-0.0234,-0.926,0.075),dp_u_dt(0.37,-0.0234,-0.926,0.075)]
    print('This no love song')


    def vectorfiled(q,t):
        w,u,p_u,p_w = q
        f = [dw_dt(w,u, p_w, p_u), du_dt(w,u,p_w,p_u),dp_w_dt(w,u,p_w,p_u),dp_u_dt(w,u,p_w,p_u)]
        return f


    def shoot(w0,u0,pu_0,pw_0, t, abserr, relerr):
        # ODE parameters
        q0 = (w0,u0,pu_0,pw_0)
        qsol = odeint(vectorfiled, q0, t, atol=abserr, rtol=relerr)
        return qsol


    def inital_conditon(r,w0,uo,pw_0,pu_0):
        phi1 = np.linspace(np.pi / 100, np.pi, 4)
        phi2 = np.linspace(np.pi / 100, np.pi, 4)
        phi3 = np.linspace(np.pi / 100, 2 * np.pi, 4)
        q=[]
        for p1 in phi1:
            w = w0 + r*np.cos(p1)
            for p2 in phi2:
                u = uo + r*np.sin(p1)*np.cos(p2)
                for p3 in phi3:
                    pw = pw_0 + r*np.sin(p1)*np.sin(p2)*np.cos(p3)
                    pu = pu_0 + r*np.sin(p1)*np.sin(p2)*np.sin(p3)
                    q.append((w,u,pw,pu))
        return q


    # ODE parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = 0.17
    numpoints = 10000

    # create the time samples for the output of the ODE solver
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    r,dt,alpha,x0=0.00001,0.00001,0.5,(lam-1)/lam
    paths,residual=[],[]
    w0, u0, pu_0, pw_0 = x0*(1-(2/lam)*epsilon**2), -(x0/lam)*epsilon, 0, 0
    wf, uf, pu_f, pw_f = 0, 0, 2*x0*epsilon, -2*np.log(lam)+(x0*(3*lam+1)/lam)*epsilon**2
    rf=(wf, uf, pu_f, pw_f)
    q0_array=inital_conditon(r,w0,u0,pw_0,pu_0)
    for q0 in q0_array:
        J = ndft.Jacobian(dq_dt)
        eigen_value,eigen_vec=la.eig(J(q0))
        postive_eig_value,postive_eig_vec=[],[]
        for e in range(np.size(eigen_value)):
            if eigen_value[e].real>0:
                postive_eig_value.append(eigen_value[e].real)
                postive_eig_vec.append(eigen_vec[:,e].reshape(4,1).real)
        current_path = shoot(q0[0]+alpha*float(postive_eig_vec[0][0])*dt+(1-alpha)*float(postive_eig_vec[1][0])*dt,q0[1]+float(alpha*postive_eig_vec[0][1])*dt+(1-alpha)*float(postive_eig_vec[1][1])*dt,q0[2]+float(postive_eig_vec[0][2])*dt+(1-alpha)*float(postive_eig_vec[1][2])*dt,q0[3]+alpha*float(postive_eig_vec[0][3])*dt+(1-alpha)*float(postive_eig_vec[1][3])*dt,t,abserr,relerr)
        paths.append(current_path)
        path_distances = find_path_distance(current_path, rf)
        residual.append(min(path_distances))
    index_of_best_path = residual.index(min(residual))
    figure(1)
    plot(t,paths[index_of_best_path][:,0])
    xlabel('t')
    ylabel('w')
    title('w vs time for hetro degree network')
    savefig('het_net_w_v_t.png',dpi=100)
    plt.show()


if __name__=='__main__':
    temp1=lambda x,y: x**2+1+y
    temp2=lambda x,y: 5*y-x
    temp5=lambda q:np.array([temp1(q[0],q[1]),temp2(q[0],q[1])])
    temp3=ndft.Jacobian(temp5)
    temp4=temp3([10,8])
    # onedshooting()
    hetro_degree_shooting()