import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import ode
import scipy.linalg as la
from scipy.integrate import odeint
from pylab import figure, plot, xlabel, grid, legend, title,savefig,ylabel
from matplotlib.font_manager import FontProperties
import csv
import functools
import numdifftools as ndft



def onedshooting(b,abserr,relerr,dt,t,r,savename):
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
        q0 = (x0, p0)
        m0 = (b, k)
        qsol = odeint(vectorfiled, q0, t, args=(m0,), atol=abserr, rtol=relerr)
        return qsol

    # create the time samples for the output of the ODE solver
    k=1.0
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
        residual.append(max(path_distances))
    index_of_best_path = residual.index(min(residual))
    figure(1)
    plot(paths[index_of_best_path][:,0],paths[index_of_best_path][:,1],linewidth=4,label='Numerical')
    theory_p=[np.log(k/(b*(1-j))) for j in paths[index_of_best_path][:,0]]
    plot(paths[index_of_best_path][:,0],theory_p,'--r',linewidth=4,label='Theory')
    plt.scatter((paths[index_of_best_path][:,0][0], paths[index_of_best_path][:,0][-1]), (paths[index_of_best_path][:,1][0], paths[index_of_best_path][:,1][-1]),c=('g', 'm'), s=(100, 100))

    xlabel('I')
    ylabel('p')
    title('Optimal path p vs I shooting')
    plt.legend()
    plt.savefig(savename+'.png', dpi=500)
    plt.show()


def hetro_degree_shooting(lam, epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,savename,hozname,vertname,titlename,plotvar,plottheory):
    Reproductive = lam/(2*(1+epsilon**2))
    ecl_dist = lambda r0, rf: np.sqrt((r0[0] - rf[0]) ** 2 + (r0[1] - rf[1]) ** 2+(r0[2] - rf[2]) ** 2+(r0[3] - rf[3]) ** 2)
    find_path_distance = lambda qsol, rf: [ecl_dist(q, rf) for q in qsol]

    #Equations of motion
    dw_dt = lambda w, u, p_w, p_u: (Reproductive*(w-u*epsilon)*((1/2)*(1-epsilon)*(-u-w+1)*np.exp((p_u+p_w)/2)+(1/2)*(epsilon+1)*(u-w+1)*np.exp((p_w-p_u)/2))
            -(1/4)*(w-u)*np.exp((p_u-p_w)/2)-(1/4)*(u+w)*np.exp((-p_u-p_w)/2))
    du_dt = lambda w, u, p_w, p_u: (Reproductive*(w-u*epsilon)*((1/2)*(1-epsilon)*(-u-w+1)*np.exp((p_u+p_w)/2)-(1/2)*(epsilon+1)*(u-w+1)*np.exp((p_w-p_u)/2))
            +(1/4)*(w-u)*np.exp((p_u-p_w)/2)-(1/4)*(u+w)*np.exp((-p_u-p_w)/2))
    dp_w_dt = lambda w, u, p_w, p_u: -(Reproductive*((1-epsilon)*(-u-w+1)*(np.exp((p_u+p_w)/2)-1)+(1+epsilon)*(u-w+1)*(np.exp((p_w-p_u)/2)-1))
            +Reproductive*(w-u*epsilon)*((1-epsilon)*(-(np.exp((p_u+p_w)/2)-1))-(epsilon+1)*(np.exp((p_w-p_u)/2)-1))
            +(1/2)*(np.exp((-p_u-p_w)/2)-1)+(1/2)*(np.exp((p_u-p_w)/2)-1))
    dp_u_dt = lambda w, u, p_w, p_u:-(Reproductive*(w-epsilon*u)*((epsilon+1)*(np.exp((p_w-p_u)/2)-1)-(1-epsilon)*(np.exp((p_u+p_w)/2)-1))
            -Reproductive*epsilon*((1-epsilon)*(-u-w+1)*(np.exp((p_u+p_w)/2)-1)+(epsilon+1)*(u-w+1)*(np.exp((p_w-p_u)/2)-1))
            +(1/2)*(np.exp((-p_u-p_w)/2)-1)+(1/2)*(1-np.exp((p_u-p_w)/2)))

    dq_dt = lambda q:np.array([dw_dt(q[0], q[1], q[2], q[3]),du_dt(q[0], q[1], q[2], q[3]),dp_w_dt(q[0], q[1], q[2], q[3]),dp_u_dt(q[0], q[1], q[2], q[3])])

    #Numerical calcuation of eq of motion
    H = lambda q: Reproductive*(q[0]-epsilon*q[1])*((1-epsilon)*(1-q[0]-q[1])*(np.exp((q[2]+q[3])/2)-1)+(1+epsilon)*(1-(q[0]-q[1]))*(np.exp((q[2]-q[3])/2)-1))+((q[0]+q[1])/2)*(np.exp(-(q[2]+q[3])/2)-1)+((q[0]-q[1])/2)*(np.exp(-(q[2]-q[3])/2)-1)
    Jacobian_H = ndft.Jacobian(H)
    dq_dt_numerical = lambda q: np.multiply(Jacobian_H(q),np.array([-1,-1,1,1]).reshape(1,4))


    def vectorfiled(q,t):
        w,u,p_w,p_u = q
        f = [dw_dt(w,u, p_w, p_u), du_dt(w,u,p_w,p_u),dp_w_dt(w,u,p_w,p_u),dp_u_dt(w,u,p_w,p_u)]
        return f

    def vectorfiled_numerical(q,t):
        w, u, p_w, p_u = q
        f = dq_dt_numerical([w,u,p_w,p_u])
        return f[0]

    def shoot(w0,u0,pw_0,pu_0, t, abserr, relerr):
        q0 = (w0, u0, pw_0, pu_0)
        qsol = odeint(vectorfiled, q0, t, atol=abserr, rtol=relerr)
        return qsol


    def inital_condtion_1d(r,w0,u0,pw_0,pu_0):
        theta=np.linspace(np.pi/100,2*np.pi,20)
        q=[]
        wi = [w0 + r * np.cos(t) for t in theta]
        # ui = [u0 + r * np.sin(t) for t in theta]
        pw_i = [pw_0 - r * np.sin(t) for t in theta]
        for w, pw in zip(wi, pw_i):
            q.append((w,u0,pw,pu_0))
        return q


    def inital_conditon(r,w0,u0,pw_0,pu_0):
        theta=np.linspace(np.pi/100,2*np.pi,20)
        q=[]
        wi = [w0 + r * np.cos(t) for t in theta]
        # ui = [u0 + r * np.sin(t) for t in theta]
        pw_i = [pw_0 - r * np.sin(t) for t in theta]
        for w, pw in zip(wi, pw_i):
            q.append((w,u0,pw,pu_0))
        return q


    def postive_eigen_vec(J,q0):
        # Find eigen vectors
        eigen_value, eigen_vec = la.eig(J(q0))
        postive_eig_vec = []
        for e in range(np.size(eigen_value)):
            if eigen_value[e].real > 0:
                postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
        return postive_eig_vec


    def plot_best_numerical_path(path,horizantal,vertical,savename,hozname,vertname,titlename,plottheory):
        figure(1)
        plot(path[:,horizantal],path[:,vertical],linewidth=4,label='Numerical')
        theory_p=[2*np.log(1/(lam*(1-j))) for j in path[:,horizantal]]
        if plottheory is True: plot(path[:,horizantal],theory_p,'--y',linewidth=4,label='Theory')
        plt.scatter((path[:,horizantal][0],path[:,horizantal][-1]),(path[:,vertical][0],path[:,vertical][-1]),c=('g','r'),s=(100,100))
        plt.legend()
        xlabel(hozname)
        ylabel(vertname)
        title(titlename)
        savefig(savename+'.png',dpi=500)
        plt.show()

    def plot_numerical_normalized_path(path,horizantal,vertical,savename,hozname,vertname,titlename,plottheory):
        figure(2)
        theory_p = [2 * np.log(1 / (lam * (1 - j))) for j in path[:, horizantal]]
        plot(path[:, horizantal], path[:, vertical]-theory_p, linewidth=4, label='Residual numerical path')
        plt.scatter((path[:,horizantal][0],path[:,horizantal][-1]),(path[:,vertical][0]-theory_p[0],path[:,vertical][-1]-theory_p[-1]),c=('g','r'),s=(100,100))
        plt.legend()
        xlabel(hozname)
        ylabel(vertname)
        title(titlename)
        savefig(savename+'_resd.png',dpi=500)
        plt.show()



    # End of deceleration of assisting functions and start of algorithm

    # Theortical start and End point of the path. and intalization
    paths,residual=[],[]
    x0 = (lam-1)/lam
    w0, u0, pu_0, pw_0 = x0*(1-(2/lam)*epsilon**2), -(x0/lam)*epsilon, 0, 0
    wf, uf, pu_f, pw_f = 0, 0, 2*x0*epsilon, -2*np.log(lam)+(x0*(3*lam+1)/lam)*epsilon**2
    rf=(wf, uf, pu_f, pw_f)

    # An array with radius r around the eq points
    q0_array=inital_condtion_1d(r,w0,u0,pw_0,pu_0)

    #The jacobian for finding the eigen vector in which to shoot
    J = ndft.Jacobian(dq_dt)

    for q0 in q0_array:
        postive_eig_vec=postive_eigen_vec(J,q0)
        for alpha in weight_of_eig_vec:
            w_i,u_i,pw_i,pu_i= q0[0]+alpha*float(postive_eig_vec[0][0])*dt+(1-alpha)*float(postive_eig_vec[1][0])*dt,q0[1]+float(alpha*postive_eig_vec[0][1])*dt+(1-alpha)*float(postive_eig_vec[1][1])*dt,q0[2]+float(postive_eig_vec[0][2])*dt+(1-alpha)*float(postive_eig_vec[1][2])*dt,q0[3]+alpha*float(postive_eig_vec[0][3])*dt+(1-alpha)*float(postive_eig_vec[1][3])*dt
            current_path=shoot(w_i,u_i,pw_i,pu_i,t,abserr,relerr)
            paths.append(current_path)
            path_distances = find_path_distance(current_path, rf)
            residual.append(max(path_distances))

    index_of_best_path = residual.index(min(residual))
    plot_best_numerical_path(paths[index_of_best_path], plotvar[0], plotvar[1], savename,hozname,vertname,titlename,plottheory)
    plot_numerical_normalized_path(paths[index_of_best_path], plotvar[0], plotvar[1], savename, hozname, vertname, titlename, plottheory)


if __name__=='__main__':
    #Network Parameters
    lam, k_avg, epsilon, sim = 1.6, 50.0, 0.0015,'h'

    # ODE parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = 40.0
    numpoints = 1000

    # Create the time samples for the output of the ODE solver
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

    # Radius around eq point,Time of to advance the self vector
    r,dt=0.0005,stoptime/ (numpoints - 1)

    # Linear combination of eigen vector vlaues for loop
    weight_of_eig_vec=np.linspace(0.0,1.0,10)

    plottheory,plotvar,titlename,hozname,vertname,savename=True,(0,2),'W vs Pw for lam=5','w','p_w','het_net_eps'

    onedshooting(lam,abserr,relerr,dt,t,r,savename) if sim=='o' else hetro_degree_shooting(lam,epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,savename,hozname,vertname,titlename,plotvar,plottheory)
