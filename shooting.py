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



def hetro_inf(beta ,gamma,epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,theta):

    def postive_eigen_vec(J,q0):
        # Find eigen vectors
        eigen_value, eigen_vec = la.eig(J(q0))
        postive_eig_vec = []
        for e in range(np.size(eigen_value)):
            if eigen_value[e].real > 0:
                postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
        return postive_eig_vec

    def vectorfiled(q, t):
        w, u, p_w, p_u = q
        f = [dy1_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
             dy2_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
             dp1_dt(float128(w), float128(u), float128(p_w), float128(p_u)),
             dp2_dt(float128(w), float128(u), float128(p_w), float128(p_u))]
        return f

    def shoot(y1_0, y2_0, p1_0, p2_0, t, abserr, relerr, J):
        q0 = (y1_0, y2_0, p1_0, p2_0)
        vect_J = lambda q, t: J(q)
        qsol = odeint(vectorfiled, q0, t, atol=abserr, rtol=relerr, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
        return qsol

    # #Numerical calcuation of eq of motion
    H = lambda q: beta*((q[0]+q[1])+epsilon*(q[0]-q[1]))*((1/2-q[0])*(np.exp(q[2])-1)+(1/2-q[1])*(np.exp(q[3])-1))+gamma*(q[0]*(np.exp(-q[2])-1)+q[1]*(np.exp(-q[3])-1))
    Jacobian_H = ndft.Jacobian(H)
    dq_dt_numerical = lambda q: np.multiply(Jacobian_H(q),np.array([-1,-1,1,1]).reshape(1,4))


    # Equations of motion
    dy1_dt = lambda y1, y2, p1, p2: -y1*gamma*np.exp(-p1) + (1/2 - y1)*beta*(y1 + y2 + (-y1 + y2)*epsilon)*np.exp(p1)
    dy2_dt = lambda y1, y2, p1, p2: (-y2)*gamma*np.exp(-p2) + (1/2 - y2)*beta*(y1 + y2 + (-y1 + y2)*epsilon)*np.exp(p2)
    dp1_dt = lambda y1, y2, p1, p2: -(gamma*(-1 + np.exp(-p1)) + beta*(y1 + y2 + (-y1 + y2)*epsilon)*(1 - np.exp(p1)) + beta*(1 - epsilon)*((1/2 - y1)*(-1 + np.exp(p1)) + (1/2 - y2)*(-1 + np.exp(p2))))
    dp2_dt = lambda y1, y2, p1, p2: -(gamma*(-1 + np.exp(-p2)) + beta*(1 + epsilon)*((1/2 - y1)*(-1 + np.exp(p1)) + (1/2 - y2)*(-1 + np.exp(p2))) + beta*(y1 + y2 + (-y1 + y2)*epsilon)*(1 - np.exp(p2)))

    dq_dt = lambda q: np.array([dy1_dt(q[0], q[1], q[2], q[3]), dy2_dt(q[0], q[1], q[2], q[3]), dp1_dt(q[0], q[1], q[2], q[3]), dp2_dt(q[0], q[1], q[2], q[3])])
    # temp=dq_dt([0.1,0.2,0.3,0.4])
    J = ndft.Jacobian(dq_dt)
    # temp_numeric= dq_dt_numerical([0.1, 0.2, 0.3, 0.4])
    y1_0, y2_0, p1_0, p2_0 = (1/2)*(1-gamma/beta), (1/2)*(1-gamma/beta), 0, 0

    def one_shot(shot_angle,lin_weight,radius=r,final_time_path=t):
        q0 = (y1_0 + radius * np.cos(shot_angle), y2_0, p1_0+radius * np.sin(shot_angle), p2_0)
        postive_eig_vec = postive_eigen_vec(J, q0)
        y1_i, y2_i, p1_i, p2_i = q0[0] + lin_weight * float(postive_eig_vec[0][0]) * dt + (
                    1 - lin_weight) * float(postive_eig_vec[1][0]) * dt \
            , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * dt + (1 - lin_weight) * float(
            postive_eig_vec[1][1]) * dt \
            , q0[2] + float(postive_eig_vec[0][2]) * dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * dt \
            , q0[3] + lin_weight * float(postive_eig_vec[0][3]) * dt + (1 - lin_weight) * float(
            postive_eig_vec[1][3]) * dt
        return shoot(y1_i, y2_i, p1_i, p2_i, final_time_path, abserr, relerr, J)


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


        #         max_time_1, max_time_2 = np.argmax(current_path[:,2]), np.argmax(current_path[:,3])
        #         if max_time_1 > max_time_2:
        #             return angle_and_lin_combo if current_path[:,2][max_time_1+1] is not 0.0 else divergence_time.append(max_time_1)
        #         else:
        #             return angle_and_lin_combo if current_path[:,3][max_time_2+1] is not 0.0 else divergence_time.append(max_time_2)
        # longest_path_time = np.argmax(divergence_time[0])
        # return divergence_values[longest_path_time]


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


    def plot_one_shot():
        path = one_shot(theta, weight_of_eig_vec)
        plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
        plt.plot(path[:, 0] + path[:, 1],
                 [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
                 linewidth=4, linestyle='--', color='y', label='Theory')
        xlabel('y1+y2')
        ylabel('p1+p2')
        title('For epsilon=0 theory vs numerical results, clancy different lambdas')
        plt.legend()
        plt.scatter((path[:, 0][0] + path[:, 1][0], path[:, 0][-1] + path[:, 1][-1]),
                    (path[:, 2][0] + path[:, 3][0], path[:, 2][-1] + path[:, 3][-1]), c=('g', 'r'), s=(100, 100))
        plt.show()

    def plot_all_var():
        path = one_shot(theta, weight_of_eig_vec)
        plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
                 linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
        plt.plot(path[:, 0] + path[:, 1],
                 [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
                 linewidth=4, linestyle='--', color='y', label='Theory')
        xlabel('y1+y2')
        ylabel('p1+p2')
        title('For epsilon=0 theory vs numerical results, clancy different lambdas')
        plt.legend()
        plt.scatter((path[:, 0][0] + path[:, 1][0], path[:, 0][-1] + path[:, 1][-1]),
                    (path[:, 2][0] + path[:, 3][0], path[:, 2][-1] + path[:, 3][-1]), c=('g', 'r'), s=(100, 100))
        plt.savefig('tot_y_v_tot_p' + '.png', dpi=500)
        plt.show()
        plt.plot(path[:, 0], path[:, 2], linewidth=4,
                 linestyle='None', Marker='.', label='y1 vs p1 for epsilon=' + str(epsilon))
        plt.plot(path[:, 1], path[:, 3], linewidth=4,
                 linestyle='--',  label='y2 vs p2 for epsilon=' + str(epsilon))
        plt.scatter((path[:, 0][0] , path[:, 0][-1] ),
                    (path[:, 2][0], path[:, 2][-1]), c=('g', 'r'), s=(100, 100))

        xlabel('Coordinate')
        ylabel('Momentum')
        title('The momentum of each group vs coordinate')
        plt.legend()
        plt.savefig('all_y_v_all_p' + '.png', dpi=500)
        plt.show()

    def plot_z():
        path = one_shot(theta, weight_of_eig_vec)
        z1=[(np.exp(-x)-1)/(1-epsilon) for x in path[:,2]]
        z2=[(np.exp(-x)-1)/(1+epsilon) for x in path[:,3]]
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
    # plot_one_shot()
    plot_all_var()
    plot_z()
    # advance_one_dt_at_time(8.0)

if __name__=='__main__':
    #Network Parameters
    lam, k_avg, epsilon, sim = 1.6, 50.0, 0.5,'h'
    # lam, k_avg, epsilon, sim = 1.6, 50.0, [0.16,0.1,0.02],'h'


    # ODE parameters22
    abserr = 1.0e-20
    relerr = 1.0e-13
    stoptime=15.88
    # stoptime = [30.272,30.709824,30.171]
    numpoints = 10000

    # Create the time samples for the output of the ODE solver
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

    dt=stoptime/(numpoints-1)
    # t,dt=[],[]
    # for s in stoptime:
    #     t.append([s * float(i) / (numpoints - 1) for i in range(numpoints)])
    #     dt.append(s/ (numpoints - 1))

    # Radius around eq point,Time of to advance the self vector
    # r=[0.019909484,0.03345353,0.163259745]
    r=0.0000089873
    theta,space=(0,2*np.pi),10
    # theta=np.linspace(np.pi/1000,2*np.pi,10)
    beta,gamma=1.6,1.0

    # Linear combination of eigen vector vlaues for loop
    weight_of_eig_vec=np.linspace(0.0,1.0,2)
    plottheory,plotvar,titlename,hozname,vertname,savename=True,(0,2),'pu vs u for epsilon=0.16','u','pu','pu_v_u_eps016_lam16'

    # onedshooting(lam,abserr,relerr,dt,t,r,savename) if sim=='o' else hetro_degree_shooting(lam,epsilon,abserr,relerr,t,r,dt,weight_of_eig_vec,savename,hozname,vertname,titlename,plotvar,plottheory,theta,space)

    # hetro_degree_shooting(lam, epsilon, abserr, relerr, t, r, dt, 1.0, savename, hozname, vertname,titlename, plotvar, plottheory, 1.5711, space)

    theta_clancy=np.linspace(0,2*np.pi,2)
    multi_r=np.linspace(0.0001,0.01,2)
    hetro_inf(beta, gamma, epsilon, abserr, relerr, t, r, dt, 0.99988218, np.pi/4-0.785084)