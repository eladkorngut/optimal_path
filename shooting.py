import numpy as np
from numpy import float128
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import ode
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
from pylab import figure, plot, xlabel, grid, legend, title,savefig,ylabel
from matplotlib.font_manager import FontProperties
import csv
import functools
import numdifftools as ndft
from cycler import cycler



from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
        qsol = odeint(vectorfiled, q0, t, args=(m0,), atol=abserr, rtol=relerr,mxstep=50000)
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


def hetro_degree_shooting(lam, epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,savename,hozname,vertname,titlename,plotvar,plottheory,theta,space):
    # Reproductive = lam/(2*(1+epsilon**2))
    # ecl_dist = lambda r0, rf: np.sqrt((r0[0] - rf[0]) ** 2 + (r0[1] - rf[1]) ** 2+(r0[2] - rf[2]) ** 2+(r0[3] - rf[3]) ** 2)
    # find_path_distance = lambda qsol, rf: [ecl_dist(q, rf) for q in qsol]
    # find_path_dis_1d = lambda qsol:[(q[1]-2*np.log(1/(lam*(1-q[0]))))**2 for q in qsol]
    #
    # #Equations of motion
    # dw_dt = lambda w, u, p_w, p_u: (Reproductive*(w-u*epsilon)*((1/2)*(1-epsilon)*(-u-w+1)*np.exp((p_u+p_w)/2)+(1/2)*(epsilon+1)*(u-w+1)*np.exp((p_w-p_u)/2))
    #         -(1/4)*(w-u)*np.exp((p_u-p_w)/2)-(1/4)*(u+w)*np.exp((-p_u-p_w)/2))
    # du_dt = lambda w, u, p_w, p_u: (Reproductive*(w-u*epsilon)*((1/2)*(1-epsilon)*(-u-w+1)*np.exp((p_u+p_w)/2)-(1/2)*(epsilon+1)*(u-w+1)*np.exp((p_w-p_u)/2))
    #         +(1/4)*(w-u)*np.exp((p_u-p_w)/2)-(1/4)*(u+w)*np.exp((-p_u-p_w)/2))
    # dp_w_dt = lambda w, u, p_w, p_u: -(Reproductive*((1-epsilon)*(-u-w+1)*(np.exp((p_u+p_w)/2)-1)+(1+epsilon)*(u-w+1)*(np.exp((p_w-p_u)/2)-1))
    #         +Reproductive*(w-u*epsilon)*((1-epsilon)*(-(np.exp((p_u+p_w)/2)-1))-(epsilon+1)*(np.exp((p_w-p_u)/2)-1))
    #         +(1/2)*(np.exp((-p_u-p_w)/2)-1)+(1/2)*(np.exp((p_u-p_w)/2)-1))
    # dp_u_dt = lambda w, u, p_w, p_u:-(Reproductive*(w-epsilon*u)*((epsilon+1)*(np.exp((p_w-p_u)/2)-1)-(1-epsilon)*(np.exp((p_u+p_w)/2)-1))
    #         -Reproductive*epsilon*((1-epsilon)*(-u-w+1)*(np.exp((p_u+p_w)/2)-1)+(epsilon+1)*(u-w+1)*(np.exp((p_w-p_u)/2)-1))
    #         +(1/2)*(np.exp((-p_u-p_w)/2)-1)+(1/2)*(1-np.exp((p_u-p_w)/2)))
    #
    # dq_dt = lambda q:np.array([dw_dt(q[0], q[1], q[2], q[3]),du_dt(q[0], q[1], q[2], q[3]),dp_w_dt(q[0], q[1], q[2], q[3]),dp_u_dt(q[0], q[1], q[2], q[3])])

    # #Numerical calcuation of eq of motion
    # H = lambda q: Reproductive*(q[0]-epsilon*q[1])*((1-epsilon)*(1-q[0]-q[1])*(np.exp((q[2]+q[3])/2)-1)+(1+epsilon)*(1-(q[0]-q[1]))*(np.exp((q[2]-q[3])/2)-1))+((q[0]+q[1])/2)*(np.exp(-(q[2]+q[3])/2)-1)+((q[0]-q[1])/2)*(np.exp(-(q[2]-q[3])/2)-1)
    # Jacobian_H = ndft.Jacobian(H)
    # dq_dt_numerical = lambda q: np.multiply(Jacobian_H(q),np.array([-1,-1,1,1]).reshape(1,4))


    # def vectorfiled(q,t):
    #     w,u,p_w,p_u = q
    #     f = [dw_dt(float128(w),float128(u), float128(p_w), float128(p_u)), du_dt(float128(w),float128(u), float128(p_w), float128(p_u)),dp_w_dt(float128(w),float128(u), float128(p_w), float128(p_u)),dp_u_dt(float128(w),float128(u), float128(p_w), float128(p_u))]
    #     # if math.isnan(f[0]) or math.isnan(f[1]) or math.isinf(f[2]) or math.isinf(3):
    #     #     stop=True
    #     return f


    # def vectorfiled_numerical(q,t):
    #     w, u, p_w, p_u = q
    #     f = dq_dt_numerical([w,u,p_w,p_u])
    #     return f[0]


    def inital_condtion_1d(r,w0,u0,pw_0,pu_0,low_theta,up_theta,space):
        theta=np.linspace(low_theta,up_theta,space)
        q=[]
        wi = [w0 + r * np.cos(t) for t in theta]
        ui = [u0 + r * np.sin(t) for t in theta]
        # pw_i = [pw_0 - r * np.sin(t) for t in theta]
        for w, u in zip(wi, ui):
            q.append((w,u,pw_0,pu_0))
        return q

    def inital_conditon(r,w0,u0,pw_0,pu_0):
        theta=np.linspace(np.pi/100,2*np.pi,20)
        q=[]
        wi = [w0 + r * np.cos(t) for t in theta]
        # ui = [u0 + r * np.sin(t) for t in theta]
        pw_i = [pw_0 - r * np.sin(t) for t in theta]
        for w, pw in zip(wi, pw_i):
            q.append((w,u0,pw_0,pu_0))
        return q


    def postive_eigen_vec(J,q0):
        # Find eigen vectors
        eigen_value, eigen_vec = la.eig(J(q0))
        postive_eig_vec = []
        for e in range(np.size(eigen_value)):
            if eigen_value[e].real > 0:
                postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
        return postive_eig_vec


    def plot_best_numerical_path(path,horizantal,vertical,savename,hozname,vertname,titlename,plottheory,qf,theortical_plot):
        figure(1)
        plot(path[:, horizantal], path[:, vertical], linewidth=4, label='Numerical', linestyle='None', Marker='.')
        theory_p=[2*np.log(1/(lam*(1-j))) for j in path[:,horizantal]]
        if plottheory is True and horizantal is 0 and vertical is 2 : plot(path[:,horizantal],theory_p,'--y',linewidth=4,label='Theory')
        if plottheory is True and horizantal is 1 and vertical is 3 :
            plot(theortical_plot[0],theortical_plot[1],'--y',linewidth=4,label='Theory')
            plt.scatter((theortical_plot[0][0],theortical_plot[0][-1]),
                        (path[:, vertical][0], path[:, vertical][-1]), c=('m', 'k'), s=(100, 100),marker='v',label='Theory start point')
        plt.scatter((path[:,horizantal][0],path[:,horizantal][-1]),(path[:,vertical][0],path[:,vertical][-1]),c=('g','r'),s=(100,100),label='Numerical start point')
        plt.legend()
        xlabel(hozname)
        ylabel(vertname)
        title(titlename)
        print('The value of x axis at the end of path: ', path[:,horizantal][-1])
        print('The value of the y axis at the end of path: ',path[:,vertical][-1],'The theory is: ',qf[vertical])
        print('The error is: ',(qf[vertical]-path[:,vertical][-1])/qf[vertical]*100,'%')
        temp_integration = simps(path[:, vertical],path[:, horizantal])
        print(0.5*temp_integration)
        s0=1/lam + np.log(lam)-1
        s1=( ( (lam-1)*(3*lam**2-10*lam-1) )/(4*math.pow(lam,3)) +(2/lam)*np.log(lam) )*epsilon**2
        theory_interation=((lam-1)**2/(2*math.pow(lam,3)))*epsilon**2 if horizantal is 1 and vertical is 3 else 1/lam + np.log(lam)-1 -( ( (lam-1)*(3*lam**2-10*lam-1) )/(4*math.pow(lam,3)) +(2/lam)*np.log(lam) )*epsilon**2
        print(theory_interation)
        print((theory_interation-0.5*temp_integration)/theory_interation*100)
        print(s0)
        print(0.5*temp_integration-s0)
        print(s1)
        savefig(savename+'.png',dpi=500)
        plt.show()

    def plot_numerical_normalized_path(path,horizantal,vertical,savename,hozname,vertname,titlename,plottheory,theortical_plot):
        figure(2)
        theory_p = [-2 * np.log((lam * (1 - j))) for j in path[:, horizantal]]
        plot(path[:, horizantal], path[:, vertical]-theory_p, linewidth=4, label='Numerical path Pw-pw(0)',linestyle='None',Marker='.')
        plt.scatter((path[:,horizantal][0],path[:,horizantal][-1]),(path[:,vertical][0]-theory_p[0],path[:,vertical][-1]-theory_p[-1]),c=('g','r'),s=(100,100),label='Start point')
        if plottheory is True: plt.plot(theortical_plot[0],theortical_plot[1],linewidth=4,label='Theory pw(1)',linestyle='--')
        plt.legend()
        xlabel(hozname)
        ylabel(vertname)
        title(titlename)
        savefig(savename+'_resd.png',dpi=500)
        plt.show()

    # def plot_multi_numerical_paths(paths,horizantal,vertical,savename,hozname,vertname,titlename,parameters):
    #     figure(3)
    #     for path in paths:
    #         plot(path[:,horizantal],path[:,vertical],linewidth=4,label=str(parameters[0])+'  '+str(par),linestyle='None',Marker='.')
    #     theory_p=[2*np.log(1/(lam*(1-j))) for j in path[:,horizantal]]
    #     if plottheory is True: plot(path[:,horizantal],theory_p,'--y',linewidth=4,label='Theory')
    #     plt.scatter((path[:,horizantal][0],path[:,horizantal][-1]),(path[:,vertical][0],path[:,vertical][-1]),c=('g','r'),s=(100,100))
    #     plt.legend()
    #     xlabel(hozname)
    #     ylabel(vertname)
    #     title(titlename)
    #     savefig(savename+'.png',dpi=500)
    #     plt.show()



    def plot_3d_path_heat(paths,x,y,z,weight):
        # create some fake data
        X, Y = np.meshgrid(x, y)
        # create the figure, add a 3d axis, set the viewing angle
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, 60)

        # here we create the surface plot, but pass V through a colormap
        # to create a different color for each patch
        ax.plot_surface(X, Y, z, facecolors=cm.Oranges(weight))


    def plot_all_paths(all_paths,horizantal,vertical,savename,hozname,vertname,titlename,parameters,w0):
        fig,ax=plt.subplots()
        for path,par in zip(all_paths,parameters):
            ax.plot(path[:,horizantal],path[:,vertical],linewidth=4,label='a='+str(par[0])
            + ',wi='+ str(round(par[1],6))+',ui='+str(round(par[2],6))+',t='+str(round(np.arccos(par[1]-w0),6)),linestyle='None',Marker='.')
        ax.scatter((path[:,horizantal][0],path[:,horizantal][-1]),
        (path[:,vertical][0],path[:,vertical][-1]),c=('g','r'),s=(100,100))
        label_params = ax.get_legend_handles_labels()
        xlabel(hozname)
        ylabel(vertname)
        title(titlename)
        savefig(savename+'.png',dpi=500)
        plt.show()
        ax.legend(loc='best')
        figl, axl = plt.subplots()
        axl.axis(False)
        axl.legend(*label_params, loc="center")
        figl.savefig("LABEL_ONLY.png")
        plt.show()

    # End of deceleration of assisting functions and start of algorithm

    # Theortical start and End point of the path. and intalization
    def plot_paths_functions():

        Reproductive = lam / (2 * (1 + epsilon ** 2))
        ecl_dist = lambda r0, rf: np.sqrt(
            (r0[0] - rf[0]) ** 2 + (r0[1] - rf[1]) ** 2 + (r0[2] - rf[2]) ** 2 + (r0[3] - rf[3]) ** 2)
        find_path_distance = lambda qsol, rf: [ecl_dist(q, rf) for q in qsol]
        find_path_dis_1d = lambda qsol: [(q[1] - 2 * np.log(1 / (lam * (1 - q[0])))) ** 2 for q in qsol]

        # Equations of motion
        dw_dt = lambda w, u, p_w, p_u: (Reproductive * (w - u * epsilon) * (
                    (1 / 2) * (1 - epsilon) * (-u - w + 1) * np.exp((p_u + p_w) / 2) + (1 / 2) * (epsilon + 1) * (
                        u - w + 1) * np.exp((p_w - p_u) / 2))
                                        - (1 / 4) * (w - u) * np.exp((p_u - p_w) / 2) - (1 / 4) * (u + w) * np.exp(
                    (-p_u - p_w) / 2))
        du_dt = lambda w, u, p_w, p_u: (Reproductive * (w - u * epsilon) * (
                    (1 / 2) * (1 - epsilon) * (-u - w + 1) * np.exp((p_u + p_w) / 2) - (1 / 2) * (epsilon + 1) * (
                        u - w + 1) * np.exp((p_w - p_u) / 2))
                                        + (1 / 4) * (w - u) * np.exp((p_u - p_w) / 2) - (1 / 4) * (u + w) * np.exp(
                    (-p_u - p_w) / 2))
        dp_w_dt = lambda w, u, p_w, p_u: -(Reproductive * (
                    (1 - epsilon) * (-u - w + 1) * (np.exp((p_u + p_w) / 2) - 1) + (1 + epsilon) * (u - w + 1) * (
                        np.exp((p_w - p_u) / 2) - 1))
                                           + Reproductive * (w - u * epsilon) * (
                                                       (1 - epsilon) * (-(np.exp((p_u + p_w) / 2) - 1)) - (
                                                           epsilon + 1) * (np.exp((p_w - p_u) / 2) - 1))
                                           + (1 / 2) * (np.exp((-p_u - p_w) / 2) - 1) + (1 / 2) * (
                                                       np.exp((p_u - p_w) / 2) - 1))
        dp_u_dt = lambda w, u, p_w, p_u: -(Reproductive * (w - epsilon * u) * (
                    (epsilon + 1) * (np.exp((p_w - p_u) / 2) - 1) - (1 - epsilon) * (np.exp((p_u + p_w) / 2) - 1))
                                           - Reproductive * epsilon * (
                                                       (1 - epsilon) * (-u - w + 1) * (np.exp((p_u + p_w) / 2) - 1) + (
                                                           epsilon + 1) * (u - w + 1) * (np.exp((p_w - p_u) / 2) - 1))
                                           + (1 / 2) * (np.exp((-p_u - p_w) / 2) - 1) + (1 / 2) * (
                                                       1 - np.exp((p_u - p_w) / 2)))

        dq_dt = lambda q: np.array(
            [dw_dt(q[0], q[1], q[2], q[3]), du_dt(q[0], q[1], q[2], q[3]), dp_w_dt(q[0], q[1], q[2], q[3]),
             dp_u_dt(q[0], q[1], q[2], q[3])])

        def vectorfiled(q, t):
            w, u, p_w, p_u = q
            f = [dw_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
                 du_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
                 dp_w_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
                 dp_u_dt(float128(w), float128(u), float128(p_w), float128(p_u))]
            # if math.isnan(f[0]) or math.isnan(f[1]) or math.isinf(f[2]) or math.isinf(3):
            #     stop=True
            return f

        low_theta, up_theta=theta[0],theta[1]
        paths=[]
        residual=[]
        x0 = (lam-1)/lam
        w0, u0, pu_0, pw_0 = x0*(1-(2/lam)*epsilon**2), -(x0/lam)*epsilon, 0, 0
        wf, uf, pu_f, pw_f = 0, 0, 2*x0*epsilon, -2*np.log(lam)+(x0*(3*lam+1)/lam)*epsilon**2
        rf=(wf, uf, pw_f, pu_f)

        w_theory_path = np.linspace(w0,wf,numpoints)
        u_theory_path = np.linspace(u0,uf,numpoints)
        pw_theory_path_first_order = [(3*(1-i)-(1+2*lam)/(lam**2)+(i*(3+i)/(lam*(1-i))))*epsilon**2 for i in w_theory_path]
        pu_theory_path_first_order = [(2*epsilon*(lam-1)/lam)*(1+(i*lam**2)/(epsilon*(lam-1))) for i in u_theory_path]

        def shoot(w0, u0, pw_0, pu_0, t, abserr, relerr, J):
            q0 = (w0, u0, pw_0, pu_0)
            vect_J = lambda q, t: J(q)
            # [qsol,temp] = odeint(vectorfiled, q0, t,atol=abserr, rtol=relerr, mxstep=10000000, hmin=1e-30,Dfun=vect_J,full_output=1)
            qsol = odeint(vectorfiled, q0, t, atol=abserr, rtol=relerr, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
            return qsol

        # An array with radius r around the eq points
        q0_array=inital_condtion_1d(r,w0,u0,pw_0,pu_0,low_theta,up_theta,space)

        #The jacobian for finding the eigen vector in which to shoot
        J = ndft.Jacobian(dq_dt)
        parameters_path=[]
        fig, ax = plt.subplots()
        for q0 in q0_array:
            postive_eig_vec=postive_eigen_vec(J,q0)
            dist_from_theory = []
            for alpha in weight_of_eig_vec:
                w_i,u_i,pw_i,pu_i= q0[0]+alpha*float(postive_eig_vec[0][0])*dt+(1-alpha)*float(postive_eig_vec[1][0])*dt,q0[1]+float(alpha*postive_eig_vec[0][1])*dt+(1-alpha)*float(postive_eig_vec[1][1])*dt,q0[2]+float(postive_eig_vec[0][2])*dt+(1-alpha)*float(postive_eig_vec[1][2])*dt,q0[3]+alpha*float(postive_eig_vec[0][3])*dt+(1-alpha)*float(postive_eig_vec[1][3])*dt
                current_path=shoot(w_i,u_i,pw_i,pu_i,t,abserr,relerr,J)
                paths.append(current_path)
                parameters_path.append((alpha,w_i,u_i,pw_i,pu_i))
                # path_distances = find_path_distance(current_path, rf)
                path_distances =find_path_dis_1d(current_path)
                residual.append(max(path_distances))
            #     residual.append(np.sum(path_distances) / ecl_dist(q0, current_path[-20]))
            #     dist_from_theory.append(
            #         la.norm(np.array([w_i, pw_i]) - np.array([w_theory_path[0], pw_theory_path_first_order[0]])))
            # ax.scatter(weight_of_eig_vec, dist_from_theory,label=str(np.arccos((q0[0]-w0)/r)))

        index_of_best_path = residual.index(min(residual))
        # plot_all_paths(paths, plotvar[0], plotvar[1], savename, hozname, vertname, titlename,parameters_path,w0)
        plot_best_numerical_path(paths[index_of_best_path], plotvar[0], plotvar[1], savename,hozname,vertname,titlename,plottheory,rf,(u_theory_path,pu_theory_path_first_order))
        # plot_numerical_normalized_path(paths[index_of_best_path], plotvar[0], plotvar[1],
        #         savename, hozname, vertname, titlename, plottheory,(w_theory_path,pw_theory_path_first_order))
        # plot_best_numerical_path(current_path, plotvar[0], plotvar[1], savename,hozname,vertname,titlename,plottheory)
        # plot_numerical_normalized_path(current_path, plotvar[0], plotvar[1], savename, hozname, vertname, titlename, plottheory)
        # print('The best path index is: ',index_of_best_path,' Alpha = ',parameters_path[index_of_best_path][0],' w_i = ',parameters_path[index_of_best_path][1],
        #       ' ui = ',parameters_path[index_of_best_path][2],' pw= ',parameters_path[index_of_best_path][3],' pu = ',parameters_path[index_of_best_path][3])
        # plt.scatter([p[0] for p in parameters_path], residual)
        # plt.scatter([np.arccos(p[1]-w0) for p in parameters_path], residual)
        # plt.scatter([np.arcsin(p[1]) for p in parameters_path], residual)
        # plt.scatter([p[0] for p in parameters_path],dist_from_theory)
        # temp2=np.gradient(current_path[:,2])
        # plt.show()
        # # print('bla',temp2[-1])
        # label_params = ax.get_legend_handles_labels()
        # xlabel('alpha')
        # ylabel('dist')
        # title('dist v alpha')
        # savefig(savename+'.png',dpi=500)
        # plt.show()
        # ax.legend(loc='best')

        # figl, axl = plt.subplots()
        # axl.axis(False)
        # axl.legend(*label_params, loc="center")
        # figl.savefig("LABEL_ONLY.png")
        # plt.show()

    # def one_path(J,q0,tf):
    #     postive_eig_vec=postive_eigen_vec(J,q0)
    #     w_i,u_i,pw_i,pu_i= q0[0]+weight_of_eig_vec*float(postive_eig_vec[0][0])*dt+(1-weight_of_eig_vec)*float(postive_eig_vec[1][0])*dt,q0[1]+float(weight_of_eig_vec*postive_eig_vec[0][1])*dt+(1-weight_of_eig_vec)*float(postive_eig_vec[1][1])*dt,q0[2]+float(postive_eig_vec[0][2])*dt+(1-weight_of_eig_vec)*float(postive_eig_vec[1][2])*dt,q0[3]+weight_of_eig_vec*float(postive_eig_vec[0][3])*dt+(1-weight_of_eig_vec)*float(postive_eig_vec[1][3])*dt
    #     current_path=shoot(w_i,u_i,pw_i,pu_i,tf,abserr,relerr,J)
    #     return current_path

    def multi_epsilons_path():
        def vectorfiled(q, t):
            w, u, p_w, p_u = q
            f = [dw_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
                 du_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
                 dp_w_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
                 dp_u_dt(float128(w), float128(u), float128(p_w), float128(p_u))]
            # if math.isnan(f[0]) or math.isnan(f[1]) or math.isinf(f[2]) or math.isinf(3):
            #     stop=True
            return f

        def shoot(w0, u0, pw_0, pu_0, t, abserr, relerr, J):
            q0 = (w0, u0, pw_0, pu_0)
            vect_J = lambda q, t: J(q)
            # [qsol,temp] = odeint(vectorfiled, q0, t,atol=abserr, rtol=relerr, mxstep=10000000, hmin=1e-30,Dfun=vect_J,full_output=1)
            qsol = odeint(vectorfiled, q0, t, atol=abserr, rtol=relerr, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
            return qsol

        x0 = (lam-1)/lam
        paths=[]
        for eps,tf,dt_path,r_path in zip(epsilon,t,dt,r):
            Reproductive = lam / (2 * (1 + eps ** 2))
            # Equations of motion
            dw_dt = lambda w, u, p_w, p_u: (Reproductive * (w - u * eps) * (
                    (1 / 2) * (1 - eps) * (-u - w + 1) * np.exp((p_u + p_w) / 2) + (1 / 2) * (eps + 1) * (
                    u - w + 1) * np.exp((p_w - p_u) / 2))
                                            - (1 / 4) * (w - u) * np.exp((p_u - p_w) / 2) - (1 / 4) * (u + w) * np.exp(
                        (-p_u - p_w) / 2))
            du_dt = lambda w, u, p_w, p_u: (Reproductive * (w - u * eps) * (
                    (1 / 2) * (1 - eps) * (-u - w + 1) * np.exp((p_u + p_w) / 2) - (1 / 2) * (eps + 1) * (
                    u - w + 1) * np.exp((p_w - p_u) / 2))
                                            + (1 / 4) * (w - u) * np.exp((p_u - p_w) / 2) - (1 / 4) * (u + w) * np.exp(
                        (-p_u - p_w) / 2))
            dp_w_dt = lambda w, u, p_w, p_u: -(Reproductive * (
                    (1 - eps) * (-u - w + 1) * (np.exp((p_u + p_w) / 2) - 1) + (1 + eps) * (u - w + 1) * (
                    np.exp((p_w - p_u) / 2) - 1))
                                               + Reproductive * (w - u * eps) * (
                                                       (1 - eps) * (-(np.exp((p_u + p_w) / 2) - 1)) - (
                                                       eps + 1) * (np.exp((p_w - p_u) / 2) - 1))
                                               + (1 / 2) * (np.exp((-p_u - p_w) / 2) - 1) + (1 / 2) * (
                                                       np.exp((p_u - p_w) / 2) - 1))
            dp_u_dt = lambda w, u, p_w, p_u: -(Reproductive * (w - eps * u) * (
                    (eps + 1) * (np.exp((p_w - p_u) / 2) - 1) - (1 - eps) * (np.exp((p_u + p_w) / 2) - 1))
                                               - Reproductive * eps * (
                                                       (1 - eps) * (-u - w + 1) * (np.exp((p_u + p_w) / 2) - 1) + (
                                                       eps + 1) * (u - w + 1) * (np.exp((p_w - p_u) / 2) - 1))
                                               + (1 / 2) * (np.exp((-p_u - p_w) / 2) - 1) + (1 / 2) * (
                                                       1 - np.exp((p_u - p_w) / 2)))

            dq_dt = lambda q: np.array(
                [dw_dt(q[0], q[1], q[2], q[3]), du_dt(q[0], q[1], q[2], q[3]), dp_w_dt(q[0], q[1], q[2], q[3]),
                 dp_u_dt(q[0], q[1], q[2], q[3])])
            J = ndft.Jacobian(dq_dt)
            w0, u0, pu_0, pw_0 = x0 * (1 - (2 / lam) * eps ** 2), -(x0 / lam) * eps, 0, 0
            q0 = (w0 + r_path * np.cos(theta), u0 + r_path * np.sin(theta), pw_0, pu_0)
            postive_eig_vec = postive_eigen_vec(J, q0)
            w_i, u_i, pw_i, pu_i = q0[0] + weight_of_eig_vec * float(postive_eig_vec[0][0]) * dt_path + (
                        1 - weight_of_eig_vec) * float(postive_eig_vec[1][0]) * dt_path, q0[1] + float(
                weight_of_eig_vec * postive_eig_vec[0][1]) * dt_path + (1 - weight_of_eig_vec) * float(
                postive_eig_vec[1][1]) * dt_path, q0[2] + float(postive_eig_vec[0][2]) * dt_path + (
                                               1 - weight_of_eig_vec) * float(postive_eig_vec[1][2]) * dt_path, q0[
                                       3] + weight_of_eig_vec * float(postive_eig_vec[0][3]) * dt_path + (
                                               1 - weight_of_eig_vec) * float(postive_eig_vec[1][3]) * dt_path
            current_path=shoot(w_i,u_i,pw_i,pu_i,tf,abserr,relerr,J)
            theory_p_1d = [2 * np.log(1 / (lam * (1 - j))) for j in current_path[:, 0][current_path[:,3]>0.001]]
            # normalized_path=[current_path[:,0],current_path[:,1]/eps,(current_path[:,2]-theory_p_1d)/eps**2,current_path[:,3]/eps]
            normalized_path = [current_path[:, 0][current_path[:,3]>0.001], current_path[:, 1][current_path[:,3]>0.001] / eps,
                               (current_path[:, 2][current_path[:,3]>0.001]- theory_p_1d) / eps ** 2, current_path[:, 3][current_path[:,3]>0.001] / eps]
            paths.append(normalized_path)
            u_theory_path = np.linspace(u0, 0.0, numpoints)/eps
            pu_theory_path_first_order = [(2 * (lam - 1) / lam) * (1 + (i * lam ** 2) / ((lam - 1)))
                                          for i in u_theory_path]
            # plt.plot(u_theory_path,pu_theory_path_first_order,linestyle='--',color='k',linewidth=4.0)
            plt.plot(normalized_path[0],normalized_path[2],linewidth=4,label='epsilon='+ str(eps))
            # plt.legend()
        w_theory_path = np.linspace(w0, 0.0, numpoints)
        pw_theory_path_first_order = [
            (3 * (1 - i) - (1 + 2 * lam) / (lam ** 2) + (i * (3 + i) / (lam * (1 - i))))  for i in
            w_theory_path]
        plt.plot(w_theory_path,pw_theory_path_first_order,linestyle='--',linewidth=4.0,label='Theory',color='k')

        plt.title('Multi epsilon, (pw-pw(0))    /epsilon^2 vs w')
        plt.xlabel('w')
        plt.ylabel('pw-pw(0)/epsilon^2')
        plt.legend()
        plt.savefig('pw_v_w_eps01_eps016_eps002_multi_eps_plot.png',dpi=500)
        plt.show()
        return paths


    # temp = multi_epsilons_path()
    plot_paths_functions()
    # print('This no love song')

def postive_eigen_vec(J,q0):
    # Find eigen vectors
    eigen_value, eigen_vec = la.eig(J(q0,None))
    postive_eig_vec = []
    for e in range(np.size(eigen_value)):
        if eigen_value[e].real > 0:
            postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
    return postive_eig_vec

def vectorfiled(q, t,dy1_dt,dy2_dt,dp1_dt,dp2_dt):
    w, u, p_w, p_u = q
    f = [dy1_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
         dy2_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
         dp1_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
         dp2_dt(float128(w), float128(u), float128(p_w), float128(p_u))]
    return f

def shoot(y1_0, y2_0, p1_0, p2_0, tshot, abserr, relerr, J,dq_dt):
    q0 = (y1_0, y2_0, p1_0, p2_0)
    vect_J = lambda q, tshot: J(q0)
    qsol = odeint(dq_dt, q0, tshot,atol=abserr, rtol=relerr, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
    return qsol


def eq_points_inf_only(epsilon,beta,gamma):
    if type(epsilon) is float:
        return (1/2)*(1-gamma/beta), (1/2)*(1-gamma/beta), 0, 0,  -np.log((epsilon + np.sqrt(
            epsilon ** 2 + (1 / 4) * ((beta / gamma) ** 2) * (1 - epsilon ** 2) ** 2) + (1 / 2) * (beta / gamma) * (
                                              1 - epsilon ** 2)) / (1 + epsilon)), -np.log((-epsilon + np.sqrt(
            epsilon ** 2 + (1 / 4) * ((beta / gamma) ** 2) * (1 - epsilon ** 2) ** 2) + (1 / 2) * (beta / gamma) * (
                                              1 - epsilon ** 2)) / (1 - epsilon))
    epsilon_lam,epsilon_mu=epsilon[0],epsilon[1]
    f_of_d = (1 / 2) * (beta / gamma) * (1 - epsilon_lam ** 2)
    D_temp_only_lam = (-1 + f_of_d + np.sqrt(epsilon_lam ** 2 + f_of_d ** 2)) / (1 - epsilon_lam ** 2)
    # d=-(beta-2*gamma-beta*epsilon_mu**2+np.sqrt(beta**2-2*(beta**2-2*gamma**2)*epsilon_mu**2+(beta**2)*(epsilon_mu**4)-4*beta*gamma*epsilon_lam*epsilon_mu*(-1+epsilon_mu**2)))/(2*gamma*(-1+epsilon_mu**2))

    d=lambda eps1,eps2: -(beta-2*gamma-beta*eps1**2+np.sqrt(beta**2-2*(beta**2-2*gamma**2)*eps1**2+(beta**2)*(eps1**4)-4*beta*gamma*eps2*eps1*(-1+eps1**2)))/(2*gamma*(-1+eps1**2))


    # d=-(beta-2*gamma-beta*epsilon_mu**2+np.sqrt(beta**2-2*(beta**2-2*gamma**2)*epsilon_mu**2+(beta**2)*(epsilon_mu**4)-4*beta*gamma*epsilon_lam*epsilon_mu*(-1+epsilon_mu**2)))/(2*gamma*(-1+epsilon_mu**2))
    # d_from_math=-((beta-2 *gamma-beta* epsilon_mu**2+np.sqrt(beta**2-2* (beta**2-2 *gamma**2)* epsilon_mu**2+beta**2 *epsilon_mu**4-4* beta* gamma* epsilon_lam *epsilon_mu *(-1+epsilon_mu**2)))/(2 *gamma* (-1+epsilon_mu**2)))
    d_for_y,d_for_p=d(epsilon_mu,epsilon_lam),d(epsilon_lam,epsilon_mu)
    # y1_0=((1-epsilon_mu)*d)/(2*(1+(1-epsilon_mu)*d))
    # y2_0=((1+epsilon_mu)*d)/(2*(1+(1+epsilon_mu)*d))
    # p1_0,p2_0=0,0
    # p1_f=-np.log(1+(1-epsilon_lam)*d)
    # p2_f=-np.log(1+(1+epsilon_lam)*d)
    y1_0=((1-epsilon_mu)*d_for_y)/(2*(1+(1-epsilon_mu)*d_for_y))
    y2_0=((1+epsilon_mu)*d_for_y)/(2*(1+(1+epsilon_mu)*d_for_y))
    p1_0,p2_0=0,0
    p1_f=-np.log(1+(1-epsilon_lam)*d_for_p)
    p2_f=-np.log(1+(1+epsilon_lam)*d_for_p)

    return y1_0,y2_0,p1_0,p2_0,p1_f,p2_f


# def eq_point_inf_sus(epsilon,beta,gamma):
#     epsilon_lam,epsilon_mu=epsilon[0],epsilon[1]
#     d=(-4*beta+2*gamma-2*gamma*epsilon_mu**2+np.sqrt(-4*(2*beta-2*gamma-2*gamma*epsilon_lam*epsilon_mu)*(2*beta-2*beta*epsilon_mu**2)+(4*beta-2*gamma+2*gamma*epsilon_mu**2)**2))/(2*(2*beta-2*beta*epsilon_mu**2))
#     y1_0=(1-epsilon_mu)/(2*(1+(1-epsilon_mu)*d))
#     y2_0=(1+epsilon_mu)/(2*(1+(1+epsilon_mu)*d))
#     p1_0,p2_0=0,0
#     p1_f=-np.log(1+(1-epsilon_mu)*d)
#     p2_f=-np.log(1+(1+epsilon_mu)*d)
#     return y1_0,y2_0,p1_0,p2_0,p1_f,p2_f


def hetro_inf(beta ,gamma,epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,theta,sample_size,shot_dq_dt):

    # #Numerical calcuation of eq of motion
    # H = lambda q: beta * ((q[0] + q[1]) + epsilon * (q[0] - q[1])) * (
    #             (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 / 2 - q[1]) * (np.exp(q[3]) - 1)) + gamma * (
    #                           q[0] * (np.exp(-q[2]) - 1) + q[1] * (np.exp(-q[3]) - 1))

    # Jacobian_H = ndft.Jacobian(H)
    # dq_dt_numerical = lambda q: np.multiply(Jacobian_H(q),np.array([-1,-1,1,1]).reshape(1,4))


    # dq_dt = lambda q: np.array(
    #     [dy1_dt(q[0], q[1], q[2], q[3]), dy2_dt(q[0], q[1], q[2], q[3]), dp1_dt(q[0], q[1], q[2], q[3]),
    #      dp2_dt(q[0], q[1], q[2], q[3])])
    # J = ndft.Jacobian(dq_dt)

    J=ndft.Jacobian(shot_dq_dt)

    # Equations of motion

    epsilon_z= epsilon if type(epsilon) is float else epsilon[0]
    z= lambda y1,y2: (beta - y1*beta - y2*beta - 2*gamma - beta*epsilon_z**2 + y1*beta*epsilon_z**2 + y2*beta*epsilon_z**2 + np.sqrt((-beta + y1*beta + y2*beta + 2*gamma + beta*epsilon_z**2 - y1*beta*epsilon_z**2 - y2*beta*epsilon_z**2)**2 -
        4*(-beta + y1*beta + y2*beta + gamma - y1*beta*epsilon_z + y2*beta*epsilon_z)*(gamma - gamma*epsilon_z**2)))/(2*(gamma - gamma*epsilon_z**2))
    # z_w_u_space=lambda w,u: (2*gamma + beta*(-1 + 2*w + epsilon**2 - (2*w*epsilon**2)*np.sqrt(4*(gamma**2)*(epsilon**2) + 8*u*beta*gamma*epsilon*(-1 + epsilon**2) + ((1 - 2*w)**2)*(beta**2)*(-1 + epsilon**2)**2)))/(2*gamma*(-1 + epsilon**2))
    # z_w_u_space = lambda w,u: -((-2*gamma + (-1 + w)*beta*(-1 + epsilon**2) + np.sqrt(4*gamma**2*epsilon**2 - 4*u*beta*gamma*epsilon*(-1 + epsilon**2) + (-1 + w)**2*beta**2*(-1 + epsilon**2)**2))/(2*gamma*(-1 + epsilon**2)))
    z_w_u_space = lambda w,u: -((-2*gamma + (-1 + 2*w)*beta*(-1 + epsilon_z**2) + np.sqrt(4*gamma**2*epsilon_z**2 - 8*u*beta*gamma*epsilon_z*(-1 + epsilon_z**2) + (1 - 2*w)**2*beta**2*(-1 + epsilon_z**2)**2))/(2*gamma*(-1 + epsilon_z**2)))
    # y1_0, y2_0, p1_0, p2_0 = (1/2)*(1-gamma/beta), (1/2)*(1-gamma/beta), 0, 0

    # p1_star_eq_motion=np.log((-2*gamma*epsilon+beta*(-1+epsilon**2)+np.sqrt(4*(gamma**2)*(epsilon**2)+(beta**2)*(-1+epsilon**2)**2))/(2*beta*epsilon*(-1+epsilon)))
    # p2_star_eq_motion=np.log((2*gamma*epsilon+beta*(-1+epsilon**2)+np.sqrt(4*(gamma**2)*(epsilon**2)+(beta**2)*(-1+epsilon**2)**2))/(2*beta*epsilon*(1+epsilon)))
    # p1_star_clancy = -np.log((epsilon + np.sqrt(
    #     epsilon ** 2 + (1 / 4) * ((beta / gamma) ** 2) * (1 - epsilon ** 2) ** 2) + (1 / 2) * (beta / gamma) * (
    #                                       1 - epsilon ** 2)) / (1 + epsilon))
    # p2_star_clancy = -np.log((-epsilon + np.sqrt(
    #     epsilon ** 2 + (1 / 4) * ((beta / gamma) ** 2) * (1 - epsilon ** 2) ** 2) + (1 / 2) * (beta / gamma) * (
    #                                       1 - epsilon ** 2)) / (1 - epsilon))

    y1_0, y2_0, p1_0, p2_0,p1_star_clancy,p2_star_clancy=eq_points_inf_only(epsilon,beta,gamma)

    def one_shot(shot_angle,lin_weight,radius=r,final_time_path=t,one_shot_dt=dt):
        q0 = (y1_0 + radius * np.cos(shot_angle), y2_0, p1_0+radius * np.sin(shot_angle), p2_0)
        postive_eig_vec = postive_eigen_vec(J, q0)
        y1_i, y2_i, p1_i, p2_i = q0[0] + lin_weight * float(postive_eig_vec[0][0]) * one_shot_dt + (
                    1 - lin_weight) * float(postive_eig_vec[1][0]) * one_shot_dt \
            , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * one_shot_dt + (1 - lin_weight) * float(
            postive_eig_vec[1][1]) * one_shot_dt \
            , q0[2] + float(postive_eig_vec[0][2]) * one_shot_dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * one_shot_dt \
            , q0[3] + lin_weight * float(postive_eig_vec[0][3]) * one_shot_dt + (1 - lin_weight) * float(
            postive_eig_vec[1][3]) * one_shot_dt
        return shoot(y1_i, y2_i, p1_i, p2_i, final_time_path, abserr, relerr, J,shot_dq_dt)


    def path_diverge(path):
        if path[:,2][np.absolute(path[:,2])>=10.0].size is not 0 or path[:,3][np.absolute(path[:,3])>=10.0].size is not 0:return True
        return False

    def when_path_diverge(path):
        p1_max_div = np.where(np.absolute(path[:, 2]) > 10.0)
        p1_div = 0.0 if not len(p1_max_div[0]) else np.where(np.absolute(path[:, 2]) > 10.0)[0][0]
        p2_max_div = np.where(np.absolute(path[:, 3]) > 10.0)
        p2_div = 0.0 if not len(p2_max_div[0]) else np.where(np.absolute(path[:, 3]) > 10.0)[0][0]
        if float(p2_div)==float(p1_div)==0.0:return 0.0
        if p1_div is 0.0: return p2_div
        if p2_div is 0.0: return p1_div
        return min(p1_div,p2_div)

    def iterate_path(shot_angle,radius,tstop,range_weight,delta_t,stop_at_infection,points_to_check_convergnce,number_of_inspection_points):
        one_shot_dt = tstop / (number_of_inspection_points - 1)
        path= one_shot(theta, (range_weight[0]+range_weight[1])/2,radius,t,one_shot_dt)
        while path[:,0][-1]+path[:,1][-1] > stop_at_infection:
            tstop=tstop+delta_t
            one_shot_dt = tstop / (number_of_inspection_points - 1)
            time_vec=np.linspace(0.0,tstop,number_of_inspection_points)
            new_range_weight = path_boundries(shot_angle,radius,time_vec,range_weight,points_to_check_convergnce,one_shot_dt)
            if new_range_weight is None:
                tstop = tstop - delta_t
                one_shot_dt = tstop / (number_of_inspection_points - 1)
                path = one_shot(theta, (range_weight[0] + range_weight[1]) / 2, radius,
                                np.linspace(0, tstop, number_of_inspection_points),one_shot_dt)
                delta_t = delta_t / 2
            else:
                range_weight = new_range_weight
                path = one_shot(theta, (range_weight[0]+range_weight[1])/2,radius,np.linspace(0,tstop,number_of_inspection_points),one_shot_dt)
        return path,range_weight,tstop


    def path_boundries(shot_angle,radius,t0,range_weight,points_to_check_convergnce,one_shot_dt):
        left,right = range_weight[0],range_weight[1]
        path_left,path_right = one_shot(shot_angle, left, radius, t0,one_shot_dt),one_shot(shot_angle, right, radius, t0,one_shot_dt)
        if path_diverge(path_left) is False and path_diverge(path_right) is False: return range_weight
        while range_weight is not None:
            path_points_to_check = np.linspace(range_weight[0], range_weight[1], points_to_check_convergnce)
            path_left, path_right = one_shot(shot_angle, left, radius, t0,one_shot_dt), one_shot(shot_angle, right, radius, t0,one_shot_dt)
            found_l,found_r=not path_diverge(path_left),not path_diverge(path_right)
            # found_l, found_r = False, False
            count = 1
            while found_l is False or found_r is False:
                if found_l is False:
                    left=path_points_to_check[count]
                    path = one_shot(shot_angle, left, radius, t0,one_shot_dt)
                    if path_diverge(path) is not True:
                        found_l=True
                        range_weight[0]=left
                if found_r is False:
                    right=path_points_to_check[-count]
                    path = one_shot(shot_angle, right, radius, t0,one_shot_dt)
                    if path_diverge(path) is not True:
                        found_r=True
                        range_weight[1]=right
                if right<=left: break
                count=count+1
            if found_r is True and found_l is True: return [left,right]
            range_weight=best_diverge_path(shot_angle,radius,t0,range_weight,points_to_check_convergnce,one_shot_dt)
        return range_weight


    def best_diverge_path(shot_angle,radius,t0,org_lin_combo,one_shot_dt):
        path=one_shot(shot_angle,org_lin_combo,radius,t0,one_shot_dt)
        org_div_time = when_path_diverge(path)
        going_up=path[:,0][int(org_div_time)-2]+path[:,1][int(org_div_time)-2]>=path[:,0][0]+path[:,0][0]
        while going_up:
            radius=radius*2
            path=one_shot(shot_angle,org_lin_combo,radius,t0,one_shot_dt)
            org_div_time = when_path_diverge(path)
            going_up = path[:, 0][int(org_div_time) - 2] + path[:, 1][int(org_div_time) - 2] >= path[:, 0][0] + path[:, 0][0]
        dl = 0.1
        path=one_shot(shot_angle,org_lin_combo,radius,t0,one_shot_dt)
        lin_combo=org_lin_combo
        while path_diverge(path) is True:
            org_div_time = when_path_diverge(path)
            lin_combo_step_up=lin_combo+dl
            lin_combo_step_down=lin_combo-dl
            path_up=one_shot(shot_angle,lin_combo_step_up,radius,t0,one_shot_dt)
            path_down=one_shot(shot_angle,lin_combo_step_down,radius,t0,one_shot_dt)
            time_div_up,time_div_down=when_path_diverge(path_up),when_path_diverge(path_down)
            if time_div_down == 0.0:
                return lin_combo_step_down,radius
            if time_div_up == 0.0:
                return lin_combo_step_up,radius
            best_time_before_diverge=max(time_div_up,time_div_down,org_div_time)
            if best_time_before_diverge == org_div_time:
                dl=dl/10
            elif best_time_before_diverge is time_div_down:
                lin_combo= lin_combo-dl
            elif best_time_before_diverge is time_div_up:
                lin_combo=lin_combo+dl if lin_combo<=1.0 else lin_combo
            path = one_shot(shot_angle,lin_combo,radius,t0,one_shot_dt)

            # if dl<1e-16 and (time_div_up==time_div_down==org_div_time):
            #     return -1
        return lin_combo,radius

    def fine_tuning(shot_angle,radius,t0,org_lin_combo,one_shot_dt):
        min_accurecy,dl=0.001,1e-4
        path = one_shot(shot_angle, org_lin_combo, radius, t0, one_shot_dt)
        pu_theory=p1_star_clancy-p2_star_clancy
        distance_from_theory = lambda p:np.sqrt(((p[:,0][-1]-p[:,1][-1])/2)**2+(p1_star_clancy-p2_star_clancy-(p[:,2][-1]-p[:,3][-1]))**2)
        # numerical_u,numerical_pu=(path[:,0][-1]-path[:,1][-1])/2,path[:,2][-1]-path[:,3][-1]
        lin_combo = org_lin_combo
        current_distance = distance_from_theory(path)
        while current_distance>min_accurecy:
            lin_combo_step_up = lin_combo + dl
            lin_combo_step_down = lin_combo - dl
            path_up = one_shot(shot_angle, lin_combo_step_up, radius, t0, one_shot_dt)
            path_down = one_shot(shot_angle, lin_combo_step_down, radius, t0, one_shot_dt)
            if not path_diverge(path_up) and distance_from_theory(path_up)<current_distance:
                current_distance=distance_from_theory(path_up)
                lin_combo=lin_combo+dl
            elif not path_diverge(path_down) and distance_from_theory(path_down)<current_distance:
                current_distance=distance_from_theory(path_down)
                lin_combo=lin_combo-dl
            elif dl<1e-16:
                break
            else:
                dl=dl/10
        return lin_combo

    def guess_path(sampleingtime,shot_angle=theta,lin_combo=weight_of_eig_vec,one_shot_dt=dt,radius=r):
        for s in sampleingtime:
            lin_combo, radius = best_diverge_path(shot_angle, radius, np.linspace(0.0,s,sample_size), lin_combo, one_shot_dt)
        lin_combo= fine_tuning(shot_angle, radius, np.linspace(0.0,sampleingtime[-1],sample_size), lin_combo, one_shot_dt)
        # plot_one_shot(shot_angle,lin_combo,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt)
        # plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,np.linspace(0.0,sampleingtime[-1],sample_size))
        return lin_combo,radius,one_shot(shot_angle, lin_combo,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt)


    def recusive_time_step(shot_angle,radius,t0,lin_combo,one_shot_dt,init_path_trunc_time):
        path_trunc_time=init_path_trunc_time
        move_forward_sim_time=init_path_trunc_time/2
        time_vec = np.linspace(0.0, path_trunc_time, numpoints)
        lin_combo,radius = best_diverge_path(shot_angle, radius, time_vec, lin_combo, one_shot_dt)
        path=one_shot(shot_angle,lin_combo,radius,t0,one_shot_dt)
        while path[:,0][-1]+path[:,1][-1]>0.10:
            path_trunc_time = move_forward_sim_time + path_trunc_time
            time_vec=np.linspace(0.0,path_trunc_time,numpoints)
            lin_combo_new,radius=best_diverge_path(shot_angle,radius,time_vec,lin_combo,one_shot_dt)
            path = one_shot(shot_angle, lin_combo_new, radius, t0, one_shot_dt)
            if lin_combo_new==-1:
                path_trunc_time = path_trunc_time-move_forward_sim_time-(3/2)*move_forward_sim_time
            elif lin_combo!=lin_combo_new:
                move_forward_sim_time=move_forward_sim_time/2
                lin_combo=lin_combo_new
            else:
                lin_combo=lin_combo_new
        path = plot_one_shot(theta, lin_combo, r, t, one_shot_dt)
        print(lin_combo)
        return path


    def shot_dt_multi(range_lin_combo,range_angle,stoping_time):
        linear_combination_vector = np.linspace(range_lin_combo[0],range_lin_combo[1],2)
        angle_vector = np.linspace(range_angle[0],range_angle[1],2)
        resmin,best_parameters=1e6,((0,0),(0,0))
        for l in range(len(linear_combination_vector)):
            for a in range(len(angle_vector)):
                angle_and_lin_combo=((linear_combination_vector[l-1],linear_combination_vector[l]),(angle_vector[a-1],angle_vector[a]))
                current_path = one_shot(linear_combination_vector[l], angle_vector[a], r, stoping_time)
                grad_path_p1 = np.gradient(current_path[:, 2], current_path[:, 0])
                grad_path_p2 = np.gradient(current_path[:, 3], current_path[:, 1])
                theory_grad = [2 / (1 - 2 * i) for i in current_path[:, 0]]
                res=[i+j-2*k for i,j,k in zip(grad_path_p1,grad_path_p2,theory_grad)]
                cum_res=np.abs(sum(res))
                if resmin>cum_res:
                    resmin=cum_res
                    best_parameters=angle_and_lin_combo
        return best_parameters


    def advance_one_dt_at_time(final_stoping_time):
        inital_time = 2.0
        range_angle=(np.pi/10,2*np.pi)
        range_lin_combo=(0,1.0)
        ranges=(range_lin_combo, range_angle)
        times_for_paths=np.linspace(inital_time,final_stoping_time,2)
        for time in times_for_paths:
            time_array = np.linspace(0.0,time,numpoints)
            ranges=shot_dt_multi(ranges[0],ranges[1],time_array)
        path = one_shot(ranges[0][0] , ranges[1][0],r,final_stoping_time)
        plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
        plt.plot(path[:, 0] + path[:, 1],
                 [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
                 linewidth=4, linestyle='--', color='y', label='Theory')
        xlabel('y1+y2')
        ylabel('p1+p2')
        title('For epsilon=0.5 theory vs numerical results, clancy different lambdas')
        plt.legend()
        plt.scatter((path[:, 0][0] + path[:, 1][0], path[:, 0][-1] + path[:, 1][-1]),
                    (path[:, 2][0] + path[:, 3][0], path[:, 2][-1] + path[:, 3][-1]), c=('g', 'r'), s=(100, 100))
        plt.show()


    def eps0():
        path=one_shot(np.pi/4,1.0)
        grad_path = np.gradient(path[:,2],path[:,0])
        theory_grad = [2/(1-2*i) for i in path[:,0]]
        plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
        plt.plot(path[:, 0] + path[:, 1],
                 [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
                 linewidth=4, linestyle='--', color='y', label='Theory')
        xlabel('y1+y2')
        ylabel('p1+p2')
        title('For epsilon=0 theory vs numerical results, clancy different lambdas')
        plt.legend()
        plt.scatter((path[:,0][0]+path[:,1][0],path[:,0][-1]+path[:,1][-1]),
        (path[:,2][0]+path[:,3][0],path[:,2][-1]+path[:,3][-1]),c=('g','r'),s=(100,100))
        savefig('clancy_eps0' + '.png', dpi=500)
        plt.show()
        plt.plot(path[:, 0][1:-1000],grad_path[1:-1000],linestyle='None',Marker='.')
        plt.plot(path[:, 0][1:-1000], theory_grad[1:-1000],linewidth=4,linestyle='--')
        plt.show()


    def plot_one_shot(angle_to_shoot=theta,linear_combination=weight_of_eig_vec,radius=r,time_vec=t,one_shot_dt=dt):
        path = one_shot(angle_to_shoot, linear_combination,radius,time_vec,one_shot_dt)
        plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
        plt.plot(path[:, 0] + path[:, 1],
                 [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
                 linewidth=4, linestyle='--', color='y', label='Theory')
        xlabel('y1+y2')
        ylabel('p1+p2')
        title('For eps='+str(epsilon)+' theory vs numerical results, clancy different lambdas')
        plt.legend()
        plt.scatter((path[:, 0][0] + path[:, 1][0], path[:, 0][-1] + path[:, 1][-1]),
                    (path[:, 2][0] + path[:, 3][0], path[:, 2][-1] + path[:, 3][-1]), c=('g', 'r'), s=(100, 100))
        plt.show()
        return path

    def plot_all_var(shot_angle=theta,lin_combo=weight_of_eig_vec,one_shot_dt=dt,radius=r,final_time_path=t):
        path = one_shot(shot_angle, lin_combo,radius,final_time_path,one_shot_dt)
        epsilon_theory = epsilon if type(epsilon) is float else epsilon[0]
        theory_path_theta1 = np.array([-np.log(1+(1-epsilon_theory)*z(x1,x2)) for x1,x2 in zip(path[:,0],path[:,1])])
        theory_path_theta2 = np.array([-np.log(1+(1+epsilon_theory)*z(x1,x2)) for x1,x2 in zip(path[:,0],path[:,1])])
        w_for_path,u_for_path=(path[:,0]+path[:,1])/2,(path[:,0]-path[:,1])/2
        pw_theory = np.array([-np.log(1+(1-epsilon_theory)*z_w_u_space(w,u))-np.log(1+(1+epsilon_theory)*z_w_u_space(w,u)) for w,u in zip(w_for_path,u_for_path)])
        pu_theory = np.array([-np.log(1+(1-epsilon_theory)*z_w_u_space(w,u))+np.log(1+(1+epsilon_theory)*z_w_u_space(w,u)) for w,u in zip(w_for_path,u_for_path)])
        f_of_d=(1/2)*(beta/gamma)*(1-epsilon_theory**2)
        D=(-1+f_of_d+np.sqrt(epsilon_theory**2+f_of_d**2))/(1-epsilon_theory**2)
        A_theory=-(1/2)*(p1_star_clancy+p2_star_clancy)-(gamma/beta)*D
        A_integration = simps(path[:, 2],path[:, 0])+simps(path[:, 3],path[:, 1])
        plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
        plt.plot(path[:, 0] + path[:, 1],
                 [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
                 linewidth=4, linestyle='--', color='y', label='Theory 1d homo')
        plt.plot(path[:, 0] + path[:, 1], theory_path_theta1+theory_path_theta2, linewidth=4,
                 linestyle=':',  label='Theory Clancy=' + str(epsilon))
        xlabel('y1+y2')
        ylabel('p1+p2')
        title('pw vs w; eps='+str(epsilon)+' Lam='+str(lam)+' Action theory='+str(round(A_theory,4))+' Action int='+str(round(A_integration,4)))
        plt.scatter((path[:, 0][0] + path[:, 1][0], path[:, 0][-1] + path[:, 1][-1]),
                    (path[:, 2][0] + path[:, 3][0], path[:, 2][-1] + path[:, 3][-1]), c=('g', 'r'), s=(100, 100))
        plt.legend()
        plt.savefig('pw_v_y' + '.png', dpi=500)
        plt.show()
        plt.plot(path[:, 0], path[:, 2], linewidth=4,
                 linestyle='None', Marker='.', label='y1 vs p1 for epsilon=' + str(epsilon))
        plt.plot(path[:, 1], path[:, 3], linewidth=4, label='y2 vs p2 for epsilon=' + str(epsilon),color='y')
        plt.plot(path[:, 0], theory_path_theta1, linewidth=4,linestyle=':',label='Theory Clancy',color='k')
        plt.plot(path[:, 1], theory_path_theta2, linewidth=4,linestyle='--',label='Theory Clancy',color='r')
        plt.scatter((path[:, 0][0] , path[:, 0][-1] ),
                    (path[:, 2][0], path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
        plt.scatter((path[:, 1][0] , path[:, 1][-1] ),
                    (path[:, 3][0], path[:, 3][-1]), c=('g', 'r'), marker='v',s=(100, 100))

        plt.scatter((0 , 0 ),
                    (p1_star_clancy, p2_star_clancy), c=('m', 'k'), s=(100, 100))
        xlabel('Coordinate')
        ylabel('Momentum')
        title('y1,y2 vs p1,p2 for epsilon='+str(epsilon)+' and Lambda='+str(lam))
        plt.legend()
        plt.savefig('p1p2_v_y1y2' + '.png', dpi=500)
        plt.show()
        plt.plot(path[:, 1]-path[:, 0], path[:, 3]-path[:, 2], linewidth=4,
                 linestyle='None', Marker='.', label='y2-y1 vs p2-p1 for epsilon=' + str(epsilon))
        plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
        plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
                    (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
        plt.scatter((y2_0-y1_0 , 0 ),
                    (0, p2_star_clancy-p1_star_clancy), c=('m', 'k'), s=(100, 100))
        xlabel('y2-y1')
        ylabel('p2-p1')
        title('p_u vs u for epsilon='+str(epsilon)+' and Lambda='+str(lam))
        plt.legend()
        plt.savefig('pu_vs_y' + '.png', dpi=500)
        plt.show()
        plt.plot(w_for_path, path[:, 2]+path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
        plt.plot(w_for_path,pw_theory,linestyle='--',linewidth=4)
        # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
        # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
        #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
        plt.scatter(((y2_0+y1_0)/2 , 0 ),(0, (p2_star_clancy+p1_star_clancy)), c=('g', 'r'), s=(100, 100))
        xlabel('w')
        ylabel('pw')
        title('p_w vs w for epsilon='+str(epsilon)+' and Lambda='+str(lam))
        plt.legend()
        plt.savefig('pw_vs_w' + '.png', dpi=500)
        plt.show()
        plt.plot(u_for_path, path[:, 2]-path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
        plt.plot(u_for_path,pu_theory,linestyle='--',linewidth=4)
        # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
        # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
        #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
        plt.scatter(((y1_0-y2_0)/2 , 0 ),
                    (0, p1_star_clancy-p2_star_clancy), c=('g', 'r'), s=(100, 100))
        xlabel('u')
        ylabel('pu')
        title('p_u vs u for epsilon='+str(epsilon)+' and Lambda='+str(lam))
        plt.legend()
        plt.savefig('pu_vs_u' + '.png', dpi=500)
        # plt.savefig('pw_vs_eps' + '.png', dpi=500)
        plt.show()


    def plot_z():
        path = one_shot(theta, weight_of_eig_vec)
        epsilon_theory=epsilon if type(epsilon) is float else epsilon[0]
        z1=[(np.exp(-x)-1)/(1-epsilon_theory) for x in path[:,2]]
        z2=[(np.exp(-x)-1)/(1+epsilon_theory) for x in path[:,3]]
        plt.plot(t,z1,linewidth=4,label='z for the 1-epsilon population')
        plt.plot(t,z2,linewidth=4,label='z for the 1+epsilon population',linestyle='--')
        plt.scatter((t[0], t[-1]),
                    (z1[0], z2[-1]), c=('g', 'r'), s=(100, 100))
        xlabel('Time')
        ylabel('z')
        title('z=(exp(-p)-1)\(1-epsilon),(exp(-p)-1)\(1+epsilon), lam=1.6 eps=0.5')
        plt.legend()
        plt.savefig('z_v_time' + '.png', dpi=500)
        plt.show()
        plt.plot(path[:,0],z1,linewidth=4,label='z for the 1-epsilon population')
        plt.plot(path[:,1],z2,linewidth=4,label='z for the 1+epsilon population',linestyle='--')
        xlabel('y')
        ylabel('z')
        title('The z=(exp(-p)-1)\(1-epsilon),(exp(-p)-1)\(1+epsilon), lam=1.6 eps=0.5')
        plt.legend()
        plt.scatter((path[:,0][0], path[:,1][-1]),
                    (z1[0], z2[-1]), c=('g', 'r'), s=(100, 100))
        plt.savefig('z_v_y' + '.png', dpi=500)
        plt.show()
        plt.plot(path[:,2],z1,linewidth=4,label='z for the 1-epsilon population')
        plt.plot(path[:,3],z2,linewidth=4,label='z for the 1+epsilon population',linestyle='--')
        plt.scatter((path[:,2][0], path[:,3][-1]),
                    (z1[0], z2[-1]), c=('g', 'r'), s=(100, 100))
        xlabel('p')
        ylabel('z')
        title('z=(exp(-p)-1)\(1-epsilon),(exp(-p)-1)\(1+epsilon), lam=1.6 eps=0.5')
        plt.legend()
        plt.savefig('z_v_p' + '.png', dpi=500)
        plt.show()


    def multi_shot_lin_angle():
        paths = []
        fig, ax = plt.subplots()
        for lin_combo in weight_of_eig_vec:
            for angle in theta:
                for radius in r:
                    current_path = one_shot(angle, lin_combo,radius)
                    paths.append(current_path)
                    ax.plot(current_path[:, 0]+current_path[1], current_path[:, 2]+current_path[3], linewidth=4, label='shot angle=' + str(round(angle, 10))
                                                                                       + ',linear weight=' + str(
                        round(lin_combo, 10))+', r='+str(round(radius,10)), linestyle='None', Marker='.')
        # ax.scatter((paths[:, 0][0], paths[:, 0][-1]),(paths[:, 1][0], paths[:, 1][-1]), c=('g', 'r'), s=(100, 100))
        label_params = ax.get_legend_handles_labels()
        xlabel('y')
        ylabel('p')
        # plt.xlim([y1_0,0.4])
        title('Multi epsilon')
        savefig('multi_shoot' + '.png', dpi=500)
        plt.show()

        figl, axl = plt.subplots()
        ax.legend(loc='best')
        axl.axis(False)
        axl.legend(*label_params, loc="center")
        figl.savefig("LABEL_ONLY.png")
        plt.show()
        return paths


    # eps0()
    # multi_shot_lin_angle()
    # path=plot_one_shot(theta,weight_of_eig_vec,r,t,dt)
    # temp_lin=best_diverge_path(theta, r, np.linspace(0.0,stoptime,numpoints), weight_of_eig_vec, dt)
    # temp_best_div,r=best_diverge_path(theta, r, t, weight_of_eig_vec, dt)
    # print(temp_best_div,' ', r)
    # path = plot_one_shot(theta, temp_best_div, r, t, dt)
    # print(when_path_diverge(path))
    plot_all_var()
    plot_z()
    # advance_one_dt_at_time(8.0)
    # temp_path,temp_range,temp_stoptime=iterate_path(theta,r,stoptime,[0.99996498,0.999965],0.1,0.013,10,numpoints)
    # print(temp_range,' , ',temp_stoptime)
    # plot_one_shot(theta,temp_range[0],r,np.linspace(0.0,temp_stoptime,numpoints))
    # plot_one_shot(0.00031416339744827493,0.9999649851242188,8.9873e-06,np.linspace(0.0,16.192,numpoints))
    # print('This no love song')
    # recusive_time_step(theta, r, t, weight_of_eig_vec, dt, 12.8)
    # temp_fine_tuning = fine_tuning(theta, r, t, weight_of_eig_vec, dt)
    # print(temp_fine_tuning)
    # temp_lin_guess,temp_radius_guess,temp_guess_path=guess_path([7,10],theta,weight_of_eig_vec,dt,r)
    # print(temp_lin_guess,' ',temp_radius_guess)
    # return temp_guess_path


if __name__=='__main__':
    #Network Parameters
    lam, k_avg, epsilon, sim = 2.0, 50.0, 0.02,'h'
    # lam, k_avg, epsilon, sim = 1.6, 50.0, [0.16,0.1,0.02],'h'


    # ODE parameters22
    abserr = 1.0e-20
    relerr = 1.0e-13
    stoptime=9.5
    # stoptime = [30.272,30.709824,30.171]
    numpoints = 10000

    # Create the time samples for the output of the ODE solver
    # t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    t = np.linspace(0.0,stoptime,numpoints)

    # dt=stoptime/(numpoints-1)
    dt=16.0/(numpoints-1)
    # t,dt=[],[]
    # for s in stoptime:
    #     t.append([s * float(i) / (numpoints - 1) for i in range(numpoints)])
    #     dt.append(s/ (numpoints - 1))

    # Radius around eq point,Time of to advance the self vector
    # r=[0.019909484,0.03345353,0.163259745]
    r=4e-8
    theta,space=(0,2*np.pi),10
    # theta=np.linspace(np.pi/1000,2*np.pi,10)
    beta,gamma=lam,1.0

    # Linear combination of eigen vector vlaues for loop
    weight_of_eig_vec=np.linspace(0.0,1.0,2)
    plottheory,plotvar,titlename,hozname,vertname,savename=True,(0,2),'pu vs u for epsilon=0.16','u','pu','pu_v_u_eps016_lam16'

    # onedshooting(lam,abserr,relerr,dt,t,r,savename) if sim=='o' else hetro_degree_shooting(lam,epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,savename,hozname,vertname,titlename,plotvar,plottheory,theta,space)

    # hetro_degree_shooting(lam, epsilon, abserr, relerr, t, r, dt, 1.0, savename, hozname, vertname,titlename, plotvar, plottheory, 1.5711, space)

    theta_clancy=np.linspace(0,2*np.pi,2)
    multi_r=np.linspace(0.0001,0.01,2)
    guessed_paths=[]
    list_of_epsilons=[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16]
    # for eps in list_of_epsilons:
    #     lamonly_dy1_dt = lambda y1, y2, p1, p2: -y1 * gamma * np.exp(-p1) + (1 / 2 - y1) * beta * (
    #                 y1 + y2 + (-y1 + y2) * eps) * np.exp(p1)
    #     lamonly_dy2_dt = lambda y1, y2, p1, p2: (-y2) * gamma * np.exp(-p2) + (1 / 2 - y2) * beta * (
    #                 y1 + y2 + (-y1 + y2) * eps) * np.exp(p2)
    #     lamonly_dp1_dt = lambda y1, y2, p1, p2: -(
    #             gamma * (-1 + np.exp(-p1)) + beta * (y1 + y2 + (-y1 + y2) * eps) * (1 - np.exp(p1)) + beta * (
    #             1 - eps) * ((1 / 2 - y1) * (-1 + np.exp(p1)) + (1 / 2 - y2) * (-1 + np.exp(p2))))
    #     lamonly_dp2_dt = lambda y1, y2, p1, p2: -(gamma * (-1 + np.exp(-p2)) + beta * (1 + eps) * (
    #             (1 / 2 - y1) * (-1 + np.exp(p1)) + (1 / 2 - y2) * (-1 + np.exp(p2))) + beta * (
    #                                                   y1 + y2 + (-y1 + y2) * eps) * (1 - np.exp(p2)))
    #     dq_dt_lamonly=lambda q,t=None:np.array([lamonly_dy1_dt(q[0],q[1],q[2],q[3]),lamonly_dy2_dt(q[0],q[1],q[2],q[3]),lamonly_dp1_dt(q[0],q[1],q[2],q[3]),lamonly_dp2_dt(q[0],q[1],q[2],q[3])])
    #     # def dq_dt_lamonly(q,t):
    #     #     return np.array([lamonly_dy1_dt(q[0],q[1],q[2],q[3]),lamonly_dy2_dt(q[0],q[1],q[2],q[3]),lamonly_dp1_dt(q[0],q[1],q[2],q[3]),lamonly_dp2_dt(q[0],q[1],q[2],q[3])])
    #     guessed_paths.append(hetro_inf(beta, gamma, eps, abserr, relerr, t, r, dt,0.9999918386580096, np.pi/4-0.785084,numpoints,dq_dt_lamonly))
    # plt.figure()

    epsilon_lam,epsilon_mu=0.5,0.5
    dy1_dt_sus_inf=lambda q: beta*((1-epsilon_lam)*q[0]+(1+epsilon_lam)*q[1])*(1-epsilon_mu)*(1/2-q[0])*np.exp(q[2])-gamma*q[0]*np.exp(-q[2])
    dy2_dt_sus_inf=lambda q: beta*((1-epsilon_lam)*q[0]+(1+epsilon_lam)*q[1])*(1+epsilon_mu)*(1/2-q[1])*np.exp(q[3])-gamma*q[1]*np.exp(-q[3])
    dtheta1_dt_sus_inf = lambda q:-beta*(1-epsilon_lam)*((1-epsilon_mu)*(1/2-q[0])*(np.exp(q[2])-1)+(1+epsilon_mu)*(1/2-q[1])*(np.exp(q[3])-1))+beta*((1-epsilon_lam)*q[0]+(1+epsilon_lam)*q[1])*(1-epsilon_mu)*(np.exp(q[2])-1)-gamma*(np.exp(-q[2])-1)
    dtheta2_dt_sus_inf = lambda q:-beta*(1+epsilon_lam)*((1-epsilon_mu)*(1/2-q[0])*(np.exp(q[2])-1)+(1+epsilon_mu)*(1/2-q[1])*(np.exp(q[3])-1))+beta*((1-epsilon_lam)*q[0]+(1+epsilon_lam)*q[1])*(1+epsilon_mu)*(np.exp(q[3])-1)-gamma*(np.exp(-q[3])-1)
    dq_dt_sus_inf = lambda q,t=None:np.array([dy1_dt_sus_inf(q),dy2_dt_sus_inf(q),dtheta1_dt_sus_inf(q),dtheta2_dt_sus_inf(q)])

    # H = lambda q: beta*((1-epsilon_lam)*q[0]+(1+epsilon_lam)*q[1])*((1-epsilon_mu)*(1/2-q[0])*(np.exp(q[2])-1)+(1+epsilon_mu)*(1/2-q[1])*(np.exp(q[3])-1))+gamma*((np.exp(-q[2])-1)*q[0]+(np.exp(-q[3])-1)*q[1])
    #
    # Jacobian_H = ndft.Jacobian(H)
    # dq_dt_numerical = lambda q: np.multiply(Jacobian_H(q),np.array([-1,-1,1,1]).reshape(1,4))
    # temp_analytical=dq_dt_numerical((0.1,0.2,0.3,0.4))
    # temp_numerical=dq_dt_sus_inf((0.1,0.2,0.3,0.4))

    hetro_inf(beta, gamma, (epsilon_lam,epsilon_mu), abserr, relerr, t, r, dt, 1.0148169762132846, np.pi / 4 - 0.785084, numpoints,
              dq_dt_sus_inf)

    # for p,eps in zip(guessed_paths,list_of_epsilons):
    #     w=(p[:,0]+p[:,1])/2
    #     pw=p[:,2]+p[:,3]
    #     linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.'])
    #     plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
    #                                cycler('linestyle', ['-', '--', ':', '-.'])))
    #     pw0=[2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(p[:, 0], p[:, 1])]
    #     plt.plot(w,(pw-pw0)/eps**2,linewidth=4,label='eps='+str(eps))
    # plt.xlabel('w')
    # plt.ylabel('(pw-pw0)/eps^2')
    # plt.title('(pw-pw0)/eps^2 vs w different epsilons')
    # plt.legend()
    # plt.savefig('pw_v_w_different_eps_normalized_lam'+str(lam)+'.png',dpi=500)
    # plt.show()
    # stoptime=15.792
    # t = np.linspace(0.0,stoptime,numpoints)
    # dt=stoptime/(numpoints-1)
    # hetro_inf(beta, gamma, epsilon, abserr, relerr, t, r, dt, 0.9999649851242188, np.pi/4-0.785084)

