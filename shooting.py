import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
from pylab import figure, plot, xlabel, grid, legend, title,savefig,ylabel
import numdifftools as ndft
from itertools import cycle



def eq_points_alpha(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    alpha,x0=epsilon_mu/epsilon_lam,(lam-1)/lam
    pu=2*x0*epsilon_lam
    u_star=-(alpha*x0)/(2*lam)*epsilon_lam
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, J=eq_hamilton_J('bc',beta,epsilon,None,gamma)
    pu_exact=p1_star_clancy-p2_star_clancy
    u_exact=(y1_0-y2_0)/2
    pw_star=-2*np.log(lam)+(1-2*alpha/lam-1/lam**2)*epsilon_lam**2
    w_star=(1/2)*x0+(1/2)*alpha*(1/lam**2)*(1-alpha*lam*x0)*epsilon_lam**2
    pw_exact=p1_star_clancy+p2_star_clancy
    w_exact=(y1_0+y2_0)/2
    w_exact_correction=w_exact-(1/2)*x0
    pw_exact_correction=pw_exact+2*np.log(lam)
    pw_theory_correction=(1-2*alpha/lam-1/lam**2)*epsilon_lam**2
    w_theory_correction=(1/2)*alpha*(1/lam**2)*(1-alpha*lam*x0)*epsilon_lam**2
    coeff=(alpha*(-1+alpha*(-1+lam)))/((-1+lam)*lam)
    pwi0=-np.log(lam*(1-2*w_star))
    coeff_full_order = (alpha * (1 + alpha - alpha * lam))/ (
                -alpha * epsilon_lam ** 2 + (-1 + lam) * (alpha * epsilon_lam) ** 2- lam * (-1 + lam))
    print('This no love song')


def eq_points_exact(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    y1star=(-2*epsilon_mu*(1 + epsilon_lam*epsilon_mu)+ lam*(-1 + epsilon_mu)*(1 + (-1 + 2*epsilon_lam)*epsilon_mu)+ np.sqrt(lam**2 +epsilon_mu*(4*epsilon_mu +lam**2*epsilon_mu*(-2 +epsilon_mu**2) +4*epsilon_lam*(lam -(-2 + lam)*epsilon_mu**2) +4*epsilon_lam**2*epsilon_mu*(lam -(-1 + lam)*epsilon_mu**2))))/(4*lam*(-1 +epsilon_lam)*(-1 +epsilon_mu)*epsilon_mu)
    y2star=(lam + epsilon_mu*(-2 + 2*lam +lam*epsilon_mu+ 2*epsilon_lam*(lam +(-1 + lam)*epsilon_mu)) -np.sqrt(lam**2 +epsilon_mu*(4*epsilon_mu +lam**2*epsilon_mu*(-2 +epsilon_mu**2) +4*epsilon_lam*(lam -(-2 + lam)*epsilon_mu**2) +4*epsilon_lam**2*epsilon_mu*(lam -(-1 + lam)*epsilon_mu**2))))/(4*lam*(1 +epsilon_lam)*epsilon_mu*(1 + epsilon_mu))
    p1star=-np.log((lam + 2*epsilon_lam -epsilon_lam**2*(lam -2*epsilon_mu) +np.sqrt(lam**2 +4*lam*epsilon_lam*epsilon_mu- 4*(-2 + lam)*epsilon_lam**3*epsilon_mu+ epsilon_lam**4*(lam**2 -4*(-1 + lam)*epsilon_mu**2) +epsilon_lam**2*(4 - 2*lam**2 + 4*lam*epsilon_mu**2)))/(2*(1 +epsilon_lam)*(1 +epsilon_lam*epsilon_mu)))
    p2star= -np.log(-(lam - 2*epsilon_lam- epsilon_lam**2*(lam + 2*epsilon_mu) +np.sqrt(lam**2 + 4*lam*epsilon_lam*epsilon_mu- 4*(-2 + lam)*epsilon_lam**3*epsilon_mu+ epsilon_lam**4*(lam**2 - 4*(-1 + lam)*epsilon_mu**2) + epsilon_lam**2*(4 - 2*lam**2 + 4*lam*epsilon_mu**2)))/(2*(-1 + epsilon_lam)*(1 + epsilon_lam*epsilon_mu)))
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def plot_generic_theory(epsilon_lam,path,x0,lam,theory_type,ax,numeric_x,case_to_run,tf,alpha):
    if theory_type is 'zw':
        u_for_path,w_for_path=numeric_x(path,epsilon_lam,alpha),(path[:, 0] + path[:, 1]) / 2
        pu_theory_clancy = np.array([-np.log(1 + (1 - epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) + np.log(
            1 + (1 + epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) for w, u in zip(w_for_path, u_for_path)])
        ax.plot(u_for_path, pu_theory_clancy, linestyle='--', linewidth=4,label='Theory clancy')
        y1_y2_and_J=eq_hamilton_J(case_to_run, beta, epsilon, tf, gamma)
        q_star=(y1_y2_and_J[0] - y1_y2_and_J[1]) / 2,0,0,y1_y2_and_J[4] - y1_y2_and_J[5]
        ax.scatter((q_star[0], q_star[1]),
                    (q_star[2], q_star[3]), c=('g', 'r'), s=(100, 100))
    elif theory_type is 'au':
        x=numeric_x(path,epsilon_lam,alpha)
        u_for_path,w_for_path=(path[:, 0] - path[:, 1]) / 2,(path[:, 0] + path[:, 1]) / 2
        pu_theory_clancy = np.array(
                [-np.log(1 + (1 - epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) + np.log(
                    1 + (1 + epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) for w, u in
                 zip(w_for_path, u_for_path)])
        ax.plot(x,pu_theory_clancy,linestyle='--', linewidth=4, label='Theory clancy')
    elif theory_type is 'wpw':
        wi = (1 / 2) * x0 + (1 / 2) * alpha * (1 / lam ** 2) * (1 - alpha * lam * x0) * epsilon_lam ** 2
        wf = 0
        pwi = 2 * np.log(lam * (1 - 2 * wi))
        pwf = (1 - 2 * alpha / lam - 1 / lam ** 2) * epsilon_lam ** 2
        w_for_path, u_for_path = (path[:, 0] + path[:, 1]) / 2, (path[:, 0] - path[:, 1]) / 2
        ax.scatter((wi, wf),
                    (pwi, pwf), c=('g', 'r'), s=(100, 100))
        pw_theory_correction = [((4 * w * alpha * lam * (1 + alpha - alpha * lam) / ((-1 + 2 * w) * (-1 + lam)) + (
                    1 - 2 * w * lam / (-1 + lam)) * (-1 - 2 * alpha * lam + lam ** 2)) / (lam ** 2)) * epsilon_lam ** 2
                                for w in w_for_path]
        pw_linear_aprox_correction=(-((1+lam)*(1+(-1+2*w_for_path)*lam))/lam**2)*epsilon_lam**2

        ax.plot(w_for_path, pw_theory_correction, linewidth=4.0, linestyle='--',label='Theory O(eps^2) eps='+str(epsilon_lam))
        ax.plot(w_for_path, pw_linear_aprox_correction, linewidth=4.0, linestyle=':',label='Linear aprox eps='+str(epsilon_lam))
    elif theory_type is 'upw0':
        w_for_path, u_for_path = (path[:, 0] + path[:, 1]) / 2, (path[:, 0] - path[:, 1]) / 2
        u_big,w_big=u_for_path/epsilon_lam,w_for_path/(1+epsilon_lam**2)
        pw_theory_second_order = ((-1 + lam * (-4 * u_big + lam - 4 * lam * w_big ** 2)) / (
                    (lam - 2 * lam * w_big) ** 2)) * epsilon_lam ** 2
        ax.plot(u_for_path,pw_theory_second_order,linewidth=4,linestyle='--',label='Thorey eps='+str(epsilon_lam))
    elif theory_type is 'uvpua':
        w_for_path, u_for_path = (path[:, 0] + path[:, 1]) / 2, (path[:, 0] - path[:, 1]) / 2
        pu_theory_clancy = np.array(
                [-np.log(1 + (1 - epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) + np.log(
                    1 + (1 + epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) for w, u in
                 zip(w_for_path, u_for_path)])
        plt.plot(u_for_path / (epsilon_lam * alpha), (pu_theory_clancy-2*x0*epsilon_lam)/epsilon_lam, linestyle='--', linewidth=4, label='Theory clancy')
    return ax


def plot_generic_theory_outside(epsilon_lam,path,x0,lam,theory_type,ax,numeric_x,case_to_run,tf,alpha):
    if theory_type is 'wpu':
        w_for_path = (path[:, 0] + path[:, 1]) / 2
        pu_theory_second_order_norm = - (2 - 2 * lam + 4 * lam * w_for_path) / (lam - 2 * w_for_path * lam)
        ax.plot(w_for_path, pu_theory_second_order_norm, linewidth=4, label='Norm approx clancy theory', linestyle=':')
        ax.scatter((w_for_path[0], w_for_path[-1]),
                   (pu_theory_second_order_norm[0], pu_theory_second_order_norm[-1]), c=('g', 'r'), s=(100, 100))
    elif theory_type is 'upw0n':
        w_for_path, u_for_path = ((path[:, 0] + path[:, 1]) / 2)/(1+epsilon_lam**2),((path[:, 0] - path[:, 1]) / 2)/epsilon_lam
        pw_theory_second_order_norm = ((-1 + lam * (-4 * u_for_path + lam - 4 * lam * w_for_path ** 2)) / (
                (lam - 2 * lam * w_for_path) ** 2))
        plt.plot(u_for_path, pw_theory_second_order_norm, linewidth=4, label='Theory', linestyle='--')
    elif theory_type is 'wpwl':
        wi = (1 / 2) * x0 + (1 / 2) * alpha * (1 / lam ** 2) * (1 - alpha * lam * x0) * epsilon_lam ** 2
        wf = 0
        pwi = 2 * np.log(lam * (1 - 2 * wi))
        pwf = (1 - 2 * alpha / lam - 1 / lam ** 2)
        ax.scatter((wi, wf),
                    (pwi, pwf), c=('g', 'r'), s=(100, 100))
        w_for_path, u_for_path = (path[:, 0] + path[:, 1]) / 2, (path[:, 0] - path[:, 1]) / 2
        pw_theory_correction = [((4 * w * alpha * lam * (1 + alpha - alpha * lam) / ((-1 + 2 * w) * (-1 + lam)) + (
                1 - 2 * w * lam / (-1 + lam)) * (-1 - 2 * alpha * lam + lam ** 2)) / (lam ** 2))
                                for w in w_for_path]

        pw_linear_aprox_correction = (-((1 + lam) * (1 + (-1 + 2 * w_for_path) * lam)) / lam ** 2)
        ax.plot(w_for_path, pw_theory_correction, linewidth=4.0, linestyle='--', label='Theory O(eps^2)')
        ax.plot(w_for_path, pw_linear_aprox_correction, linewidth=4.0, linestyle=':', label='Linear aprox')
    elif theory_type is 'uvpua':
        u_theory = np.linspace(-x0 / (2 * lam), 0, 100)
        pu_theory = 4 * lam * u_theory
        ax.plot(u_theory, pu_theory, linestyle=':', linewidth=4, label='Theory approx O(eps^2)')
    elif theory_type is 'wvu':
        wi, wf = (1 / 2) * x0, 0
        w_for_u = np.linspace(wi, wf, 100) / (1 + epsilon_lam ** 2)
        u_v_w_theory = [x * ((1 + (-1 + 2 * x) * lam) * (-2 - lam + 2 * x * (1 + lam))) / (2 * lam) for x in w_for_u]
        ax.plot(np.linspace(wi, wf, 100), u_v_w_theory, linewidth=4, linestyle='--', label='Theory')
    elif theory_type is 'puiu':
        pu_for_path=path[:,2]-path[:,3]
        u_function_pu=-(pu_for_path*(-2*epsilon_lam*(-1 + lam) +pu_for_path*lam)*(pu_for_path*lam-2*epsilon_lam*(1 + 2*lam)))/(4*(pu_for_path - 2*epsilon_lam)**3*lam**3)
        # integral_result_theory = epsilon_lam * (((lam - 1) ** 3) / (4 * lam ** 3))
        integral_result_norm_theory = ((lam - 1) ** 3) / (4 * lam ** 3)
        ax.plot(pu_for_path / epsilon_lam, u_function_pu, linestyle='--', linewidth=4,
                 label='Theory, I=' + str(round(integral_result_norm_theory, 4)))
    return ax


def generic_plot(guessed_paths,list_of_epsilons,numeric_x,numeric_y,theory_type,
                 x_label,y_label,x0,lam,name_title,savename,case_to_run,tf,labeladdon=lambda x,y:''):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for path,epsilon in zip(guessed_paths,list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        alpha = epsilon_mu / epsilon_lam
        ax.plot(numeric_x(path,epsilon_lam,alpha), numeric_y(path,epsilon_lam,x0), linewidth=4,
                 linestyle='None', Marker='.', label='Numeric eps=' + str(epsilon)+labeladdon(path,epsilon_lam))
        ax=plot_generic_theory(epsilon_lam,path,x0,lam,theory_type,ax,numeric_x,case_to_run,tf,alpha)
    ax=plot_generic_theory_outside(epsilon_lam,path,x0,lam,theory_type,ax,numeric_x,case_to_run,tf,alpha)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(name_title+' lam='+ str(beta))
    ax.legend()
    plt.tight_layout()
    fig.savefig(savename + '.png', dpi=500)
    plt.show()



def plot_multi_eps_theory(epsilon,lam,theory_type,ax,q_star):
    epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
    delta_mu=1-epsilon_mu
    if theory_type is 'p2':
        y2_for_theory = np.linspace(q_star[1], 0, 10000)
        p2_theory = ((-1 + epsilon_lam) * (y2_for_theory * lam + (-1 + 4 * y2_for_theory) * (
                    1 + (-1 + y2_for_theory) * lam) * epsilon_lam)) / (
                                (-1 + 2 * y2_for_theory) * (1 + epsilon_lam) * (-lam + (-2 + lam) * epsilon_lam))
        ax.plot(y2_for_theory, p2_theory, linewidth=4, label='Theory', linestyle='--')
    elif theory_type is 'p1':
        y1_for_theory = np.linspace(q_star[0], 0, 10000) / delta_mu
        p1_theory = ((1 + 4 * (y1_for_theory) - lam) * np.log(2 - lam + (2 * (-1 + lam)) / (1 + epsilon_lam))) / (
                    -1 + lam)
        ax.plot(y1_for_theory, p1_theory, linewidth=4, label='Theory', linestyle='--')
    elif theory_type is 'pl1':
        y1_for_theory=np.linspace(q_star[0],0,10000)
        p1_theory= ((1 - lam +2*y1_for_theory*(-2 + lam -2/(-1 +epsilon_mu))))/2
        ax.plot(y1_for_theory, p1_theory, linewidth=4, label='Theory', linestyle='--')
    elif theory_type is 'pl2':
        y2_for_theory=np.linspace(q_star[1],0,10000)
        p2_theory = ((-1 + epsilon_mu)*(-1 + lam - 2*y2_for_theory*lam + (4*y2_for_theory*epsilon_mu)/((-1 + 2*y2_for_theory)*(-lam + (-2 + lam)*epsilon_mu))))/(2*(1 + epsilon_mu))
        ax.plot(y2_for_theory, p2_theory, linewidth=4, label='Theory', linestyle='--')
    return ax


def plot_deltas(guessed_paths,list_of_epsilons,axis_x,axis_y,sim,graph_fig,
                 x_label,y_label,lam,name_title,savename,labeladdon=lambda x,y:''):
    for sim_paths,sim_epsilons in zip(guessed_paths,list_of_epsilons):
        for theory_type,numeric_x,numeric_y,x_label_c,y_label_c,name_title_c,savename_c in zip(graph_fig,axis_x,axis_y,x_label,y_label,name_title,savename):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for path,epsilon in zip(sim_paths,sim_epsilons):
                epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
                y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, J = eq_hamilton_J(sim, lam, epsilon,
                                                                                                         None, 1.0)
                q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
                ax.plot(numeric_x(path,epsilon,lam), numeric_y(path,epsilon,lam), linewidth=4,
                         linestyle='None', Marker='.', label='Numeric eps=' + str(epsilon)+labeladdon(path,epsilon_lam))
                # ax=plot_multi_eps_theory(epsilon,lam,theory_type,ax,q_star)
            ax = plot_multi_eps_theory(epsilon, lam, theory_type, ax, q_star)
            # ax=plot_generic_theory_outside(epsilon_lam,path,x0,lam,theory_type,ax,numeric_x,case_to_run,tf,alpha)
            ax.set_xlabel(x_label_c)
            ax.set_ylabel(y_label_c)
            ax.set_title(name_title_c+' lam='+ str(lam))
            ax.legend()
            plt.tight_layout()
            fig.savefig(savename_c + '.png', dpi=500)
            plt.show()


def plot_integation(guessed_paths,list_of_epsilons,lam,sim):
    fig_tot,fig_i1,fig_i2 = plt.figure(),plt.figure(),plt.figure()
    ax_tot,ax_i1,ax_i2 = fig_tot.add_subplot(1, 1, 1),fig_i1.add_subplot(1, 1, 1),fig_i2.add_subplot(1, 1, 1)
    s0= 1 / lam - 1 + np.log(lam)
    for sim_paths,sim_epsilons in zip(guessed_paths,list_of_epsilons):
        action_numeric,I1_numeric,I2_numeric,delta_mu,delta_lam=[],[],[],[],[]
        for path, epsilon in zip(sim_paths, sim_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            I1, I2 = simps(path[:, 2], path[:, 0]), simps(path[:, 3], path[:, 1])
            p2_numerical_normalized = (path[:,3] + np.log(lam*(1-2*path[:,1])))
            I2_numerical_correction = simps(p2_numerical_normalized, path[:,1])
            action_numeric.append(I1+I2-s0/2)
            I1_numeric.append(I1)
            I2_numeric.append(I2_numerical_correction)
            delta_mu.append(1-epsilon_mu)
            delta_lam.append(1-epsilon_lam)

    # delta_mu,delta_lam=np.array(delta_mu),np.array(delta_lam)
    # delta_mu_theory=np.linspace(min(delta_mu),max(delta_mu),1000)
    # action_theory = (delta_mu_theory*((-1 + lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)) - (2* (-1 + epsilon_lam)*(lam*(-1 + lam - lam*np.log(lam)) + (-3 + (5 - 2*lam)*lam + (-2 + lam)*lam*np.log(lam))*epsilon_lam))/(lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam))))/8
    # I1_theory = (((-1 + lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)))/ 8)*delta_mu_theory
    # I2_theory = (((-1 + epsilon_lam)*(lam*(1 - lam + lam*np.log(lam)) + (3 - 5*lam + 2*lam**2 - (-2 + lam)*lam*np.log(lam))*epsilon_lam))/(4*lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam)))*delta_mu_theory

    if sim is'dem':
        delta=np.array(delta_mu)
        delta_theory = np.linspace(min(delta), max(delta), 1000)
        action_theory = (delta_theory*((-1 + lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)) - (2* (-1 + epsilon_lam)*(lam*(-1 + lam - lam*np.log(lam)) + (-3 + (5 - 2*lam)*lam + (-2 + lam)*lam*np.log(lam))*epsilon_lam))/(lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam))))/8
        I1_theory = (((-1 + lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)))/ 8)*delta_theory
        I2_theory = (((-1 + epsilon_lam)*(lam*(1 - lam + lam*np.log(lam)) + (3 - 5*lam + 2*lam**2 - (-2 + lam)*lam*np.log(lam))*epsilon_lam))/(4*lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam)))*delta_theory
        label_for_x='delta_mu'
    elif sim is 'del':
        delta=np.array(delta_lam)
        delta_theory = np.linspace(min(delta), max(delta), 1000)
        action_theory = ((delta_theory *(-1 +epsilon_mu)*((-1 + lam) ** 2 *lam +(3 - 4 * lam +lam ** 2 +2 * lam * np.log(lam))*epsilon_mu))/ (4 *lam *(1 +epsilon_mu)*(-lam +(-2 + lam) *epsilon_mu)))
        I1_theory = delta_theory*(((-1 + lam)**2*(-1 + epsilon_mu))/(-8*lam + 8*(-2 + lam)*epsilon_mu))
        I2_theory = delta_theory*(-((-1 + epsilon_mu)*(-((-1 + lam)**2*lam) + ((-1 + lam)*(6 + (-3 + lam)*lam)- 4*lam*np.log(lam))*epsilon_mu))/(8*lam*(1 + epsilon_mu)*(-lam + (-2 + lam)*epsilon_mu)))
        label_for_x='delta_lam'


    ax_i1.plot(delta,I1_numeric,linewidth=4,linestyle='None',markersize=10,Marker='o',label='Numeric')
    ax_i2.plot(delta,I2_numeric,linewidth=4,linestyle='None',markersize=10,Marker='o',label='Numeric')
    ax_tot.plot(delta,action_numeric,linewidth=4,linestyle='None',markersize=10,Marker='o',label='Numeric')

    ax_i1.plot(delta_theory,I1_theory,linewidth=4,linestyle='--',label='Theory')
    ax_i2.plot(delta_theory,I2_theory,linewidth=4,linestyle='--',label='Theory')
    ax_tot.plot(delta_theory,action_theory,linewidth=4,linestyle='--',label='Theory')

    ax_i1.set_xlabel(label_for_x)
    ax_i1.set_ylabel('I1')
    ax_i1.set_title('I1 vs delta' + ' lam=' + str(lam))
    ax_i1.legend()
    ax_i2.set_xlabel(label_for_x)
    ax_i2.set_ylabel('I2')
    ax_i2.set_title('I2 vs delta' + ' lam=' + str(lam))
    ax_i2.legend()
    ax_tot.set_xlabel(label_for_x)
    ax_tot.set_ylabel('s1')
    ax_tot.set_title('s1 vs delta' + ' lam=' + str(lam))
    ax_tot.legend()
    plt.tight_layout()
    fig_tot.savefig('action_correction' + '.png', dpi=200)
    fig_i1.savefig('i1' + '.png', dpi=200)
    fig_i2.savefig('i2' + '.png', dpi=200)
    plt.show()

def plot_integration_theory_z(guessed_paths, list_of_epsilons,sim,beta,gamma,graph_type):
    fig_tot, fig_correction = plt.figure(), plt.figure()
    ax_tot, ax_correction = fig_tot.add_subplot(1, 1, 1), fig_correction.add_subplot(1, 1, 1)
    lam = beta / gamma
    # s0 = 1 / lam - 1 + np.log(lam)
    for sim_paths, sim_epsilons, s in zip(guessed_paths, list_of_epsilons, sim):
        action_numeric, action_numeric_correction, eps_mu_array, eps_lam_array = [], [], [], []
        for path, epsilon in zip(sim_paths, sim_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, J = eq_hamilton_J('lm', beta,
                                                                                                     epsilon_lam,
                                                                                                     t, gamma)

            y1_0_linear, y2_0_linear, p1_0_linear, p2_0_linear, p1_star_clancy_linear, p2_star_clancy_linear, dq_dt_sus_inf_linear, J = eq_hamilton_J(
                s, beta, epsilon,
                t, gamma)
            y1_for_linear = np.linspace(path[:, 0][-1], 0, 1000)
            py1_linear = p1_star_clancy_linear - (
                        (p1_star_clancy_linear - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
            y2_for_linear = np.linspace(path[:, 1][-1], 0, 1000)
            py2_linear = p1_star_clancy_linear - (
                        (p1_star_clancy_linear - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
            I_addition_to_path = simps(py1_linear, y1_for_linear) + simps(py2_linear, y2_for_linear)

            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            eps_mu_array.append(epsilon_mu)
            eps_lam_array.append(epsilon_lam)
            f_of_d = (1 / 2) * (beta / gamma) * (1 - epsilon_lam ** 2)
            D = (-1 + f_of_d + np.sqrt(epsilon_lam ** 2 + f_of_d ** 2)) / (1 - epsilon_lam ** 2)
            A_theory_clancy = -(1 / 2) * (q_star[2] + q_star[3]) - (gamma / beta) * D
            A_integration = simps(path[:, 2], path[:, 0]) + simps(path[:, 3], path[:, 1]) + I_addition_to_path
            action_numerical_correction = A_integration - A_theory_clancy
            action_numeric.append(A_integration)
            action_numeric_correction.append(action_numerical_correction)
        if graph_type is 's':
            eps_lam_array = np.array(eps_lam_array)
            action_numeric_correction = np.array(action_numeric_correction)
            # eps_lam_theory = np.linspace(min(eps_lam_array), max(eps_lam_array), 1000)
            # action_theory = np.array([action_o1_epslam(eps_lam, epsilon_mu, lam) for eps_lam in eps_lam_theory])
            ax_tot.plot(eps_lam_array, action_numeric, linewidth=4, linestyle='None', markersize=10, Marker='o',
                        label='epsilon=' + str(epsilon_mu))
            ax_correction.plot(eps_lam_array, action_numeric_correction / epsilon_mu, linewidth=4, linestyle='None',
                               markersize=10,
                               Marker='o', label='epsilon=' + str(epsilon_mu))
            # ax_correction.plot(eps_lam_theory, action_theory, linewidth=4,
            #                    label='Theory')

    eps_mu_array = np.array(eps_mu_array)
    eps_lam_array = np.array(eps_lam_array)
    if graph_type is 'm':
        eps_mu_theory = np.linspace(min(eps_mu_array), max(eps_mu_array), 1000)
        action_theory = np.array([action_o1_epslam(epsilon_lam, eps_mu, lam) for eps_mu in eps_mu_theory])
        action_slope_numerical = np.polyfit(eps_mu_array, action_numeric_correction, 1)
        ax_tot.plot(eps_mu_array, action_numeric, linewidth=4, linestyle='None', markersize=10, Marker='o',
                    label='Numeric')
        ax_correction.plot(eps_mu_array, action_numeric_correction, linewidth=4, linestyle='None', markersize=10,
                           Marker='o', label='Numeric, slope= ' + str(round(action_slope_numerical[0], 4)))
        ax_correction.plot(eps_mu_theory, action_theory, linewidth=4,
                           label='Theory, slope= ' + str(round(action_o1_epslam(epsilon_lam, 1.0, lam), 4)))
        ax_tot.set_xlabel('eps_mu')
        ax_tot.set_ylabel('A')
        ax_tot.set_title('Total action vs eps_mu' + ' lam=' + str(lam))
        ax_tot.legend()
        plt.tight_layout()
        fig_tot.savefig('action_total_with_clancy' + '.png', dpi=200)
        ax_correction.set_xlabel('epsilon_mu')
        ax_correction.set_ylabel('S(1)')
        ax_correction.set_title('S(1) (action minus clacny theorm) vs eps_mu' + ' lam=' + str(lam))
        ax_correction.legend()
        plt.tight_layout()
        fig_correction.savefig('action_correction_epslam_' + str(epsilon_lam).replace('.', '') + '.png', dpi=200)
        plt.show()
    elif graph_type is 'l':
        eps_lam_theory = np.linspace(min(eps_lam_array), max(eps_lam_array), 1000)
        action_theory = np.array([action_o1_epslam(eps_lam, epsilon_mu, lam) for eps_lam in eps_lam_theory])
        ax_tot.plot(eps_lam_array, action_numeric, linewidth=4, linestyle='None', markersize=10, Marker='o',
                    label='Numeric')
        ax_correction.plot(eps_lam_array, action_numeric_correction, linewidth=4, linestyle='None', markersize=10,
                           Marker='o', label='Numeric')
        ax_correction.plot(eps_lam_theory, action_theory, linewidth=4,
                           label='Theory')

        # ax_tot.plot(delta_theory,action_theory,linewidth=4,linestyle='--',label='Theory')

        ax_tot.set_xlabel('eps_lam')
        ax_tot.set_ylabel('A')
        ax_tot.set_title('Total action vs eps_lam' + ' lam=' + str(lam) + ' epsilon_mu= ' + str(epsilon_mu))
        ax_tot.legend()
        plt.tight_layout()
        fig_tot.savefig('action_total_with_clancy' + '.png', dpi=200)

        ax_correction.set_xlabel('epsilon_lam')
        ax_correction.set_ylabel('S(1)')
        ax_correction.set_title('S(1) vs eps_lam' + ' lam=' + str(lam) + ' epsilon_mu= ' + str(epsilon_mu))
        ax_correction.legend()
        plt.tight_layout()
        fig_correction.savefig('action_correction_epsmu_' + str(epsilon_mu).replace('.', '') + '.png', dpi=200)
        plt.show()
    elif graph_type is 's':
        eps_lam_theory = np.linspace(0.0, 1.0, 1000)
        action_theory = np.array([action_o1_epslam_norm(eps_lam, lam) for eps_lam in eps_lam_theory])
        ax_correction.plot(eps_lam_theory, action_theory, linewidth=4, linestyle='--', label='Theory')
        ax_tot.set_xlabel('eps_lam')
        ax_tot.set_ylabel('A')
        ax_tot.set_title('Total action vs eps_lam' + ' lam=' + str(lam))
        ax_tot.legend()
        plt.tight_layout()
        fig_tot.savefig('action_total_with_clancy' + '.png', dpi=200)

        ax_correction.set_xlabel('epsilon_lam')
        ax_correction.set_ylabel('S(1)')
        ax_correction.set_title('S(1) vs eps_lam' + ' lam=' + str(lam))
        ax_correction.legend()
        plt.tight_layout()
        fig_correction.savefig('action_correction_epsmu_multi_mu_lam' + str(lam).replace('.', '') + '.png', dpi=200)
        plt.show()



def eq_points_inf_only(epsilon,beta,gamma):
    if type(epsilon) is float:
        return (1/2)*(1-gamma/beta), (1/2)*(1-gamma/beta), 0, 0,  -np.log((epsilon + np.sqrt(
            epsilon ** 2 + (1 / 4) * ((beta / gamma) ** 2) * (1 - epsilon ** 2) ** 2) + (1 / 2) * (beta / gamma) * (
                                              1 - epsilon ** 2)) / (1 + epsilon)), -np.log((-epsilon + np.sqrt(
            epsilon ** 2 + (1 / 4) * ((beta / gamma) ** 2) * (1 - epsilon ** 2) ** 2) + (1 / 2) * (beta / gamma) * (
                                              1 - epsilon ** 2)) / (1 - epsilon))
    d=lambda eps1,eps2: -(beta-2*gamma-beta*eps1**2+np.sqrt(beta**2-2*(beta**2-2*gamma**2)*eps1**2+(beta**2)*(eps1**4)-4*beta*gamma*eps2*eps1*(-1+eps1**2)))/(2*gamma*(-1+eps1**2))
    d_for_y,d_for_p=d(epsilon[1],epsilon[0]),d(epsilon[0],epsilon[1])
    return ((1-epsilon[1])*d_for_y)/(2*(1+(1-epsilon[1])*d_for_y)),((1+epsilon[1])*d_for_y)/(2*(1+(1+epsilon[1])*d_for_y)),0.0,0.0,-np.log(1+(1-epsilon[0])*d_for_p),-np.log(1+(1+epsilon[0])*d_for_p)


z_y1_y2 = lambda y1, y2,epsilon,beta,gamma: (
                               beta - y1 * beta - y2 * beta - 2 * gamma - beta * epsilon ** 2 + y1 * beta * epsilon ** 2 + y2 * beta * epsilon ** 2 + np.sqrt(
                           (
                                       -beta + y1 * beta + y2 * beta + 2 * gamma + beta * epsilon ** 2 - y1 * beta * epsilon ** 2 - y2 * beta * epsilon ** 2) ** 2 -
                           4 * (
                                       -beta + y1 * beta + y2 * beta + gamma - y1 * beta * epsilon + y2 * beta * epsilon) * (
                                       gamma - gamma * epsilon ** 2))) / (2 * (gamma - gamma * epsilon ** 2))
z_w_u_space = lambda w, u,epsilon,beta,gamma: -((-2 * gamma + (-1 + 2 * w) * beta * (-1 + epsilon ** 2) + np.sqrt(
    4 * gamma ** 2 * epsilon ** 2 - 8 * u * beta * gamma * epsilon * (-1 + epsilon ** 2) + (
                1 - 2 * w) ** 2 * beta ** 2 * (-1 + epsilon ** 2) ** 2)) / (2 * gamma * (-1 + epsilon ** 2)))
z_eps_mu = lambda y1, y2,epsilon_lam,epsilon_mu,lam:(2 +2*epsilon_lam*epsilon_mu + lam*(-1 + epsilon_lam**2)*(1 - y1 - y2 + (y1 - y2)*epsilon_mu)- np.sqrt((-1 + y1 + y2)**2*lam**2 + epsilon_lam*(4*(y1 - y2)*lam + epsilon_lam*(4 - 2*(-1 + y1 + y2)**2*lam**2 + lam*epsilon_lam*(-4*y1 + 4*y2 + (-1 + y1 + y2)**2*lam*epsilon_lam))) - 2*((1 - y1 - y2)*lam + (2 + (-1 + y1 + y2)*lam)*epsilon_lam**2)*((-y1 + y2)*lam + epsilon_lam*(-2 + (y1 - y2)*lam*epsilon_lam))*epsilon_mu + ((y1 - y2)**2*lam**2 - 2*lam*(2*(-1 + y1 + y2) + (y1 - y2)**2*lam)*epsilon_lam**2 + (4 + 4*(-1 + y1 + y2)*lam + (y1 - y2)**2*lam**2)*epsilon_lam**4)*epsilon_mu**2))/(2*(-1 + epsilon_lam**2)*(1 + epsilon_lam*epsilon_mu))

y1_path_clancy = lambda p1,p2,epsilon_mu,lam: (np.exp(p1) *(-1 +epsilon_mu)+ np.exp(2*p1)*lam *(-1 +epsilon_mu)** 2 +np.exp(p2)*(1 +epsilon_mu) - np.sqrt(np.exp(2*p1) *(-1 +epsilon_mu) ** 2 +2 *np.exp(p1 +p2)*(-1 +epsilon_mu ** 2) +np.exp(2 *p2)*((1 +epsilon_mu) ** 2 +np.exp(2 *p1) * lam ** 2 *(-1 +epsilon_mu ** 2) ** 2))) /(2 * np.exp(p1) *lam * (np.exp(p1) *(-1 +epsilon_mu) ** 2 +np.exp(p2)*(-1 +epsilon_mu ** 2)))

y2_path_clancy = lambda p1,p2,epsilon_mu,lam: (-(np.exp(p1)*(-1 + epsilon_mu)) + np.exp(p2)*(1 + epsilon_mu)*(-1 + np.exp(p2)*lam*(1 + epsilon_mu)) - np.sqrt(np.exp(2*p1)*(-1 + epsilon_mu)**2 + 2*np.exp(p1 + p2)*(-1 + epsilon_mu**2) + np.exp(2*p2)*((1 + epsilon_mu)**2 + np.exp(2*p1)*lam**2*(-1 + epsilon_mu**2)**2)))/(2*np.exp(p2)*lam*(np.exp(p2)*(1 + epsilon_mu)**2 + np.exp(p1)*(-1 + epsilon_mu**2)))

y1_path_clancy_z = lambda z,epsilon_mu,epsilon_lam,lam: ((1 + z - z*epsilon_lam)*((-1 + epsilon_mu)/(1 + z - z*epsilon_lam) + (lam*(-1 + epsilon_mu)**2)/(1 + z - z*epsilon_lam)**2 + (1 + epsilon_mu)/(1 + z + z*epsilon_lam) - np.sqrt((lam**2 + 4*z**2*epsilon_lam**2 - 8*z*(1 + z)*epsilon_lam*epsilon_mu + (4*(1 + z)**2 - 2*lam**2)*epsilon_mu**2 + lam**2*epsilon_mu**4)/((1 + z)**2 - z**2*epsilon_lam**2)**2)))/(2*lam*((-1 + epsilon_mu)**2/(1 + z - z*epsilon_lam) + (-1 + epsilon_mu**2)/(1 + z + z*epsilon_lam)))

y2_path_clancy_z = lambda z,epsilon_mu,epsilon_lam,lam:((1 + z + z*epsilon_lam)*((1 - epsilon_mu)/(1 + z - z*epsilon_lam) + ((1 + epsilon_mu)*(-1 - z + lam - z*epsilon_lam + lam*epsilon_mu))/(1 + z + z*epsilon_lam)**2 - np.sqrt((lam**2 + 4*z**2*epsilon_lam**2 - 8*z*(1 + z)*epsilon_lam*epsilon_mu + (4*(1 + z)**2 - 2*lam**2)*epsilon_mu**2 + lam**2*epsilon_mu**4)/((1 + z)**2 - z**2*epsilon_lam**2)**2)))/(2*lam*((1 + epsilon_mu)**2/(1 + z + z*epsilon_lam) + (-1 + epsilon_mu**2)/(1 + z - z*epsilon_lam)))

action_o1_epslam = lambda epsilon_lam,epsilon_mu,lam: (epsilon_lam*(-2 + lam + np.log(4) - 2*np.log(lam - lam*epsilon_lam**2 + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)) - lam*epsilon_lam**2+ np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4))*epsilon_mu)/(2*lam*(-1 + epsilon_lam**2))

action_o1_epslam_norm = lambda epsilon_lam,lam: (epsilon_lam*(-2 + lam + np.log(4) - 2*np.log(lam - lam*epsilon_lam**2 + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)) - lam*epsilon_lam**2+ np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)))/(2*lam*(-1 + epsilon_lam**2))


def eq_point_alpha(epsilon,beta,gamma):
    # what I need to return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy
    epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
    lam=beta/gamma
    alpha = epsilon_mu / epsilon_lam
    x0=1-1/lam
    y1star = x0/2+((alpha-alpha*lam)/(2*lam**2))*epsilon_lam-((alpha*(1+alpha)*(-1+lam)/(2*lam**2)))*epsilon_lam**2
    y2star = x0/2+(alpha*(-1+lam)/(2*lam**2))*epsilon_lam-(alpha*(1+alpha)*(-1+lam)/(2*lam**2))*epsilon_lam**2
    p1star = -np.log(lam)+((-1+lam)/lam)*epsilon_lam+((-1+lam)*(1+lam+2*alpha*lam)/(2*lam**2))*epsilon_lam**2
    p2star = -np.log(lam)+(-1+1/lam)*epsilon_lam+((-1+lam)*(1+lam+2*alpha*lam)/(2*lam**2))*epsilon_lam**2
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_point_alpha_reverse(epsilon,beta,gamma):
    # what I need to return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy
    epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
    alpha = epsilon_lam / epsilon_mu
    lam=beta/gamma
    x0=1-1/lam
    y1star = x0/2+((1-lam)/(2*lam**2))*epsilon_mu-((1+alpha)*(-1+lam)/(2*lam**2))*epsilon_mu**2
    y2star = x0/2+((-1+lam)/(2*lam**2))*epsilon_mu-((1+alpha)*(-1+lam)/(2*lam**2))*epsilon_mu**2
    p1star = -np.log(lam)+(alpha*(-1+lam)/lam)*epsilon_mu+(alpha*(-1+lam)*(alpha+(2+alpha)*lam)/(2*lam**2))*epsilon_mu**2
    p2star = -np.log(lam)+alpha*(-1+1/lam)*epsilon_mu+(alpha*(-1+lam)*(alpha+(2+alpha)*lam)/(2*lam**2))*epsilon_mu**2
    return y1star, y2star, 0.0, 0.0, p1star, p2star



def eq_point_delta_eps_lam_const(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    delta_lam,delta_mu=1-epsilon_lam,1-epsilon_mu
    alpha,x0 = delta_mu / delta_lam,1-1/lam
    y1star=(alpha*(-1 + lam)*delta_lam)/4- (alpha**2*(-2 + lam)*(-1 + lam)*delta_lam**2)/8
    y2star=(-1 + lam)/(2*lam) +(alpha*(-1 + lam)*delta_lam**2)/(8*lam)
    p1star= ((1 - lam)*delta_lam)/2 + ((-3 + lam)*(-1 + lam)*delta_lam**2)/8
    p2star= -np.log(lam) + ((alpha - alpha*lam)*delta_lam**2)/4
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_point_delta_eps_mu_const(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    delta_lam,delta_mu=1-epsilon_lam,1-epsilon_mu
    alpha,x0 = delta_lam / delta_mu,1-1/lam
    y1star=((-1 + lam)*delta_mu)/4 - ((-2 + lam)*(-1 + lam)*delta_mu**2)/8
    y2star= (-1 + lam)/(2*lam) +(alpha*(-1 + lam)*delta_mu**2)/(8*lam)
    p1star= ((alpha - alpha*lam)*delta_mu)/2+ (alpha**2*(-3 + lam)*(-1 + lam)*delta_mu**2)/8
    p2star= -np.log(lam) + ((alpha - alpha*lam)*delta_mu**2)/4
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_point_delta_mu(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    delta_lam,delta_mu=1-epsilon_lam,1-epsilon_mu
    y1star= ((-1 + lam)*delta_mu)/4- ((-1 + lam)*delta_mu**2*(-1 +(-1 + lam)*epsilon_lam))/(4*(1 +epsilon_lam))
    y2star = (-1 + lam)/(2*lam) -((-1 + lam)*delta_mu*(-1 +epsilon_lam))/(4*lam*(1 +epsilon_lam)) + ((-1 + lam)**2*delta_mu**2*(-1 +epsilon_lam))/(8*lam*(1 +epsilon_lam))
    p1star = -np.log((lam -(-2 + lam)*epsilon_lam)/(1 +epsilon_lam))- ((-1 + lam)*lam*delta_mu*(-1 +epsilon_lam)**2*epsilon_lam)/((1 +epsilon_lam)*(lam -(-2 + lam)*epsilon_lam)**2) -((-1 + lam)*lam*delta_mu**2*(-1 +epsilon_lam)**2*epsilon_lam**2*(lam*(3 + lam) -2*(-3 + lam)*lam*epsilon_lam+ (8 - 9*lam +lam**2)*epsilon_lam**2))/(2*(1 +epsilon_lam)**2*(lam -(-2 + lam)*epsilon_lam)**4)
    p2star = -np.log(lam) - ((-1 + lam)*delta_mu*(-1 + epsilon_lam)*epsilon_lam)/(-lam + epsilon_lam*(-2 + (-2 + lam)*epsilon_lam)) - ((-1 + lam)*delta_mu**2*(-1 + epsilon_lam)*epsilon_lam**2*(lam*(3 + lam) + epsilon_lam*(2 - 2*(-2 + lam)*lam + (-6 + lam)*(-1 + lam)*epsilon_lam)))/(2*(1 + epsilon_lam)**2*(-lam + (-2 + lam)*epsilon_lam)**3)
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_point_delta_lam(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    delta_lam,delta_mu=1-epsilon_lam,1-epsilon_mu
    p1star = (1/2)*(1-lam)*delta_lam
    p2star = -np.log(lam) + ((-1+lam)*(-1+epsilon_mu)/(2*(1+epsilon_mu)))*delta_lam
    y1star = -((-1 + lam)*lam*delta_lam*(-1 + epsilon_mu)**2*epsilon_mu)/(2*(-lam + (-2 + lam)*epsilon_mu)**3) + ((-1 + lam)*(-1 + epsilon_mu))/(-2*lam + 2*(-2 + lam)*epsilon_mu)
    y2star = (-1 + lam)/(2.*lam) + ((-1 + lam)*delta_lam*(-1 + epsilon_mu)*epsilon_mu)/(2*lam*(1 + epsilon_mu)*(-lam + (-2 + lam)*epsilon_mu))
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_point_eps_mu(epsilon,beta,gamma):
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    y1star= (-1 + lam)/(2*lam) - ((-1 + lam)*(1 + epsilon_lam)*epsilon_mu)/(2*lam**2)
    y2star= (-1 + lam)/(2*lam) - ((-1 + lam)*(-1 + epsilon_lam)*epsilon_mu)/(2*lam**2)
    p1star= -np.log((lam + 2*epsilon_lam - lam*epsilon_lam**2 + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4))/(2*(1 + epsilon_lam))) - (lam*epsilon_lam*(-1 + epsilon_lam**2)*(-2 + lam - lam*epsilon_lam**2 + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4))*epsilon_mu)/(np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)*(lam + 2*epsilon_lam - lam*epsilon_lam**2 + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)))
    p2star= np.log((-2*(-1 + epsilon_lam))/(lam - epsilon_lam*(2 + lam*epsilon_lam) +np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 +lam**2*epsilon_lam**4))) -(lam*epsilon_lam*(-1 + epsilon_lam**2)*(-2 + lam - lam*epsilon_lam**2 +np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 +lam**2*epsilon_lam**4))*epsilon_mu)/(np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 +lam**2*epsilon_lam**4)*(lam - epsilon_lam*(2 + lam*epsilon_lam) +np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 +lam**2*epsilon_lam**4)))
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_hamilton_J(case_to_run,beta,epsilon=0.0,t=None,gamma=1.0):

    def bimodal_mu_lam(lam=beta):
        epsilon_lam,epsilon_mu=epsilon[0],epsilon[1]
        dy1_dt_sus_inf = lambda q: lam * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 - epsilon_mu) * (
                    1 / 2 - q[0]) * np.exp(q[2]) - gamma * q[0] * np.exp(-q[2])
        dy2_dt_sus_inf = lambda q: lam * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 + epsilon_mu) * (
                    1 / 2 - q[1]) * np.exp(q[3]) - gamma * q[1] * np.exp(-q[3])
        dtheta1_dt_sus_inf = lambda q: -lam * (1 - epsilon_lam) * (
                    (1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 + epsilon_mu) * (1 / 2 - q[1]) * (
                        np.exp(q[3]) - 1)) + lam * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (
                                                   1 - epsilon_mu) * (np.exp(q[2]) - 1) - gamma * (np.exp(-q[2]) - 1)
        dtheta2_dt_sus_inf = lambda q: -lam * (1 + epsilon_lam) * (
                    (1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 + epsilon_mu) * (1 / 2 - q[1]) * (
                        np.exp(q[3]) - 1)) + lam * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (
                                                   1 + epsilon_mu) * (np.exp(q[3]) - 1) - gamma * (np.exp(-q[3]) - 1)
        dq_dt_sus_inf = lambda q, t=None: np.array(
            [dy1_dt_sus_inf(q), dy2_dt_sus_inf(q), dtheta1_dt_sus_inf(q), dtheta2_dt_sus_inf(q)])
        return dq_dt_sus_inf

    if case_to_run is '1d':
        # Hamilton eq for 1d case
        d1_dx_dt = lambda i, p: beta * (1 - i) * i * np.exp(p) - gamma * i * np.exp(-p)
        d1_dp_dt = lambda i, p: beta * (2 * i - 1) * (np.exp(p) - 1) - gamma * (np.exp(p) - 1)
        d1_dq_dt = lambda q, t=None: [d1_dx_dt(q[0], q[1]), d1_dp_dt(q[0], q[1])]

        # eq point 1d case
        d1_x0, d1_p0, d1_xf, d1_pf = 1 - gamma / beta, 0, 0, np.log(gamma / beta)
        return d1_x0, d1_p0, d1_xf, d1_pf,d1_dq_dt,ndft.Jacobian(d1_dq_dt)
    elif case_to_run is 'he':
        # Hamilton eq hetro degree miki's paper
        Reproductive = beta / (2 * (1 + epsilon ** 2))
        hetro_dw_dt = lambda w, u, p_w, p_u: (Reproductive * (w - u * epsilon) * (
                (1 / 2) * (1 - epsilon) * (-u - w + 1) * np.exp((p_u + p_w) / 2) + (1 / 2) * (epsilon + 1) * (
                u - w + 1) * np.exp((p_w - p_u) / 2))
                                              - (1 / 4) * (w - u) * np.exp((p_u - p_w) / 2) - (1 / 4) * (
                                                          u + w) * np.exp(
                    (-p_u - p_w) / 2))
        hetro_du_dt = lambda w, u, p_w, p_u: (Reproductive * (w - u * epsilon) * (
                (1 / 2) * (1 - epsilon) * (-u - w + 1) * np.exp((p_u + p_w) / 2) - (1 / 2) * (epsilon + 1) * (
                u - w + 1) * np.exp((p_w - p_u) / 2))
                                              + (1 / 4) * (w - u) * np.exp((p_u - p_w) / 2) - (1 / 4) * (
                                                          u + w) * np.exp(
                    (-p_u - p_w) / 2))
        hetro_dp_w_dt = lambda w, u, p_w, p_u: -(Reproductive * (
                (1 - epsilon) * (-u - w + 1) * (np.exp((p_u + p_w) / 2) - 1) + (1 + epsilon) * (u - w + 1) * (
                np.exp((p_w - p_u) / 2) - 1))
                                                 + Reproductive * (w - u * epsilon) * (
                                                         (1 - epsilon) * (-(np.exp((p_u + p_w) / 2) - 1)) - (
                                                         epsilon + 1) * (np.exp((p_w - p_u) / 2) - 1))
                                                 + (1 / 2) * (np.exp((-p_u - p_w) / 2) - 1) + (1 / 2) * (
                                                         np.exp((p_u - p_w) / 2) - 1))
        hetro_dp_u_dt = lambda w, u, p_w, p_u: -(Reproductive * (w - epsilon * u) * (
                (epsilon + 1) * (np.exp((p_w - p_u) / 2) - 1) - (1 - epsilon) * (np.exp((p_u + p_w) / 2) - 1))
                                                 - Reproductive * epsilon * (
                                                         (1 - epsilon) * (-u - w + 1) * (
                                                             np.exp((p_u + p_w) / 2) - 1) + (
                                                                 epsilon + 1) * (u - w + 1) * (
                                                                     np.exp((p_w - p_u) / 2) - 1))
                                                 + (1 / 2) * (np.exp((-p_u - p_w) / 2) - 1) + (1 / 2) * (
                                                         1 - np.exp((p_u - p_w) / 2)))

        hetro_dq_dt = lambda q, t=None: np.array(
            [hetro_dw_dt(q[0], q[1], q[2], q[3]), hetro_du_dt(q[0], q[1], q[2], q[3]),
             hetro_dp_w_dt(q[0], q[1], q[2], q[3]),
             hetro_dp_u_dt(q[0], q[1], q[2], q[3])])

        # miki's paper approximation for the action
        theory_hetro_action_u = ((beta - 1) ** 2 / (2 * math.pow(beta, 3))) * epsilon ** 2
        theory_hetro_action_w = 1 / beta + np.log(beta) - 1 - (((beta - 1) * (3 * beta ** 2 - 10 * beta - 1))
                                / (4 * math.pow(beta, 3)) + (2 / beta) * np.log(beta)) * epsilon ** 2

        # Miki's paper hetro degree eq points
        x0 = (beta - 1) / beta
        w0, u0, pu_0, pw_0 = x0 * (1 - (2 / beta) * epsilon ** 2), -(x0 / beta) * epsilon, 0, 0
        wf, uf, pu_f, pw_f = 0, 0, 2 * x0 * epsilon, -2 * np.log(beta) + (x0 * (3 * beta + 1) / beta) * epsilon ** 2
        return w0, u0, pu_0, pw_0,wf, uf, pu_f, pw_f,hetro_dq_dt,ndft.Jacobian(hetro_dq_dt)

    elif case_to_run is 'lm':
        # Hamiltonian both sus and inf from clancy
        H = lambda q: beta * ((q[0] + q[1]) + epsilon * (q[0] - q[1])) * (
                (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 / 2 - q[1]) * (np.exp(q[3]) - 1)) + gamma * (
                              q[0] * (np.exp(-q[2]) - 1) + q[1] * (np.exp(-q[3]) - 1))

        lamonly_dy1_dt = lambda y1, y2, p1, p2: -y1 * gamma * np.exp(-p1) + (1 / 2 - y1) * beta * (
                    y1 + y2 + (-y1 + y2) * epsilon) * np.exp(p1)
        lamonly_dy2_dt = lambda y1, y2, p1, p2: (-y2) * gamma * np.exp(-p2) + (1 / 2 - y2) * beta * (
                    y1 + y2 + (-y1 + y2) * epsilon) * np.exp(p2)
        lamonly_dp1_dt = lambda y1, y2, p1, p2: -(
                gamma * (-1 + np.exp(-p1)) + beta * (y1 + y2 + (-y1 + y2) * epsilon) * (1 - np.exp(p1)) + beta * (
                1 - epsilon) * ((1 / 2 - y1) * (-1 + np.exp(p1)) + (1 / 2 - y2) * (-1 + np.exp(p2))))
        lamonly_dp2_dt = lambda y1, y2, p1, p2: -(gamma * (-1 + np.exp(-p2)) + beta * (1 + epsilon) * (
                (1 / 2 - y1) * (-1 + np.exp(p1)) + (1 / 2 - y2) * (-1 + np.exp(p2))) + beta * (
                                                      y1 + y2 + (-y1 + y2) * epsilon) * (1 - np.exp(p2)))
        dq_dt_lamonly=lambda q,t=None:np.array([lamonly_dy1_dt(q[0],q[1],q[2],q[3]),lamonly_dy2_dt(q[0],q[1],q[2],q[3]),lamonly_dp1_dt(q[0],q[1],q[2],q[3]),lamonly_dp2_dt(q[0],q[1],q[2],q[3])])
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_inf_only(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy,dq_dt_lamonly,ndft.Jacobian(dq_dt_lamonly)
    elif case_to_run is 'bc':
        dq_dt_sus_inf=bimodal_mu_lam()
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_inf_only(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'al':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_alpha(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'la':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_alpha_reverse(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'x':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_exact(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'dl':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_delta_eps_lam_const(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'dm':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_delta_eps_mu_const(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'dem':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_delta_mu(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'del':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_delta_lam(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    elif case_to_run is 'mu':
        dq_dt_sus_inf = bimodal_mu_lam(beta/(1+epsilon[0]*epsilon[1]))
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_point_eps_mu(epsilon, beta, gamma)
        return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)
    return None



def postive_eigen_vec(J,q0):
    # Find eigen vectors
    eigen_value, eigen_vec = la.eig(J(q0,None))
    postive_eig_vec = []
    for e in range(np.size(eigen_value)):
        if eigen_value[e].real > 0:
            postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
    return postive_eig_vec


def shoot(y1_0, y2_0, p1_0, p2_0, tshot, J,dq_dt):
    q0 = (y1_0, y2_0, p1_0, p2_0)
    vect_J = lambda q, tshot: J(q0)
    qsol = odeint(dq_dt, q0, tshot,atol=1.0e-20, rtol=1.0e-13, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
    return qsol


def one_shot(shot_angle,lin_weight,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt):
    q0 = (q_star[0] + radius * np.cos(shot_angle), q_star[1], 0+radius * np.sin(shot_angle), 0)
    postive_eig_vec = postive_eigen_vec(J, q0)
    y1_i, y2_i, p1_i, p2_i = q0[0] + lin_weight * float(postive_eig_vec[0][0]) * one_shot_dt + (
                1 - lin_weight) * float(postive_eig_vec[1][0]) * one_shot_dt \
        , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][1]) * one_shot_dt \
        , q0[2] + float(postive_eig_vec[0][2]) * one_shot_dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * one_shot_dt \
        , q0[3] + lin_weight * float(postive_eig_vec[0][3]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][3]) * one_shot_dt
    return shoot(y1_i, y2_i, p1_i, p2_i, final_time_path,J,shot_dq_dt)

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


def change_shot_angle(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    path = one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    going_up = path[:,0][-10]+path[:,1][-10]>=path[:,0][0]+path[:,1][0]
    dtheta=1e-3
    shot_angle_down,shot_angle_up=shot_angle - dtheta,shot_angle + dtheta
    count,max_steps=0,int(np.pi/dtheta)
    while going_up and max_steps>count:
        path_down=one_shot(shot_angle_down,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        if not path_down[:, 0][-10] + path_down[:, 1][-10] >= path_down[:, 0][0] + path_down[:, 1][0]:
            shot_angle=shot_angle_down
            break
        path_up=one_shot(shot_angle_up,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        if not path_up[:, 0][-10] + path_up[:, 1][-10] >= path_up[:, 1][0] + path_up[:, 0][0]:
            shot_angle=shot_angle_up
            break
        shot_angle_down, shot_angle_up = shot_angle_down - dtheta, shot_angle_up + dtheta
        count=count+1
    return shot_angle


def path_going_up(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    org_radius=radius
    path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    org_div_time = when_path_diverge(path)
    going_up = path[:,0][int(org_div_time)-2]+path[:,1][int(org_div_time)-2]>=path[:,0][0]+path[:,1][0]
    while going_up:
        radius=radius*2
        path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        org_div_time = when_path_diverge(path)
        going_up = path[:,0][int(org_div_time)-2]+path[:,1][int(org_div_time)-2]>=path[:,0][0]+path[:,1][0]
    if radius>5e-3:
        shot_angle=change_shot_angle(shot_angle,org_radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt)
        radius=org_radius
    path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    return path,radius,shot_angle


def best_diverge_path(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    path,radius,shot_angle=path_going_up(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt)
    org_div_time = when_path_diverge(path)
    dl,lin_combo = 0.1,org_lin_combo
    while path_diverge(path) is True:
        org_div_time = when_path_diverge(path)
        lin_combo_step_up=lin_combo+dl
        lin_combo_step_down=lin_combo-dl
        path_up=one_shot(shot_angle,lin_combo_step_up,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        path_down=one_shot(shot_angle,lin_combo_step_down,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        time_div_up,time_div_down=when_path_diverge(path_up),when_path_diverge(path_down)
        if time_div_down == 0.0:
            return lin_combo_step_down,radius,shot_angle
        if time_div_up == 0.0:
            return lin_combo_step_up,radius,shot_angle
        best_time_before_diverge=max(time_div_up,time_div_down,org_div_time)
        if best_time_before_diverge == org_div_time:
            dl=dl/10
        elif best_time_before_diverge is time_div_down:
            lin_combo= lin_combo-dl
        elif best_time_before_diverge is time_div_up:
            lin_combo=lin_combo+dl
        path = one_shot(shot_angle,lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    return lin_combo,radius,shot_angle

def fine_tuning(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt):
    min_accurecy,dl=0.001,1e-4
    path = one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    distance_from_theory = lambda p:np.sqrt(((p[:,0][-1]-p[:,1][-1])/2)**2+(q_star[2]-q_star[3]-(p[:,2][-1]-p[:,3][-1]))**2)
    lin_combo = org_lin_combo
    current_distance = distance_from_theory(path)
    if path[:,0][-1]+path[:,1][-1]<1e-1:
        while current_distance>min_accurecy:
            lin_combo_step_up = lin_combo + dl
            lin_combo_step_down = lin_combo - dl
            path_up = one_shot(shot_angle, lin_combo_step_up, q_star,radius, final_time_path,one_shot_dt,J,shot_dq_dt)
            path_down = one_shot(shot_angle, lin_combo_step_down,q_star ,radius, final_time_path, one_shot_dt,J,shot_dq_dt)
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

def guess_path(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,org_radius,sample_size,J,shot_dq_dt):
    radius=org_radius
    for s in sampleingtime:
        lin_combo, radius,shot_angle = best_diverge_path(shot_angle, radius,lin_combo,one_shot_dt,q_star,np.linspace(0.0,s,sample_size),J,shot_dq_dt )
    lin_combo, radius,shot_angle = best_diverge_path(shot_angle, org_radius, lin_combo, one_shot_dt, q_star,np.linspace(0.0, sampleingtime[-1], sample_size), J, shot_dq_dt)
    lin_combo= fine_tuning(shot_angle, lin_combo,q_star,radius, np.linspace(0.0,sampleingtime[-1],sample_size), one_shot_dt,J,shot_dq_dt)
    # plot_one_shot(shot_angle,lin_combo,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt)
    # plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,np.linspace(0.0,sampleingtime[-1],sample_size))
    return lin_combo,radius,shot_angle, one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt,J,shot_dq_dt)

def guess_path_lam(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,org_radius,sample_size,J,shot_dq_dt,beta):
    radius=org_radius
    for s in sampleingtime:
        lin_combo, radius,shot_angle = best_diverge_path(shot_angle, radius,lin_combo,one_shot_dt,q_star,np.linspace(0.0,s,sample_size),J,shot_dq_dt)
        lin_combo = fine_tuning(shot_angle, lin_combo, q_star, radius, np.linspace(0.0, s, sample_size),
                                one_shot_dt, J, shot_dq_dt)
        path=one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,s,sample_size),one_shot_dt,J,shot_dq_dt)
        if path[:,0][-1]+path[:,1][-1]<1e-3:break
    # lin_combo, radius = best_diverge_path(shot_angle, org_radius, lin_combo, one_shot_dt, q_star,np.linspace(0.0, sampleingtime[-1], sample_size), J, shot_dq_dt)
    # lin_combo= fine_tuning(shot_angle, lin_combo,q_star,radius, np.linspace(0.0,sampleingtime[-1],sample_size), one_shot_dt,J,shot_dq_dt)
    # plot_one_shot(shot_angle,lin_combo,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt)
    # plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,np.linspace(0.0,sampleingtime[-1],sample_size))
    return lin_combo,radius, one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,s,sample_size),one_shot_dt,J,shot_dq_dt)


def multi_eps_normalized_path(case_to_run,list_of_epsilons,beta,gamma,numpoints,one_shot_dt,radius,lin_combo=1.00008204478397,org_shot_angle=np.pi/4-0.785084):
    guessed_paths=[]
    shot_angle=org_shot_angle
    if type(beta) is list:
        for l in beta:
            sampleingtime = [6.0,7.0, 9.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
                             16.5, 17.0, 17.5, 18.0, 18.5, 19.5, 20.0]
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt, J = eq_hamilton_J(case_to_run, l,
                                                                                                  list_of_epsilons, t, gamma)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            lin_combo, temp_radius, path = guess_path_lam(sampleingtime, np.pi / 4 - 0.785084, lin_combo, q_star,
                                                      one_shot_dt, radius, numpoints, J, shot_dq_dt,l)
            guessed_paths.append(path)
    else:
        for eps in list_of_epsilons:
            sampleingtime=[7.0,9.0,10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.5,20.0]
            # sampleingtime = np.linspace(7.0,40.0,10)
            # sampleingtime=[7.0,9.0,10.0]
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt,J = eq_hamilton_J(case_to_run,beta,eps,t,gamma)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            # lam=beta if type(beta) is float else beta/(1+eps[0]*eps[1])
            lin_combo,temp_radius,shot_angle,path=guess_path(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,radius,numpoints,J,shot_dq_dt)
            # lin_combo,temp_radius,path=guess_path(sampleingtime,np.pi/4-0.74,lin_combo,q_star,one_shot_dt,radius,numpoints,J,shot_dq_dt)
            guessed_paths.append(path)

    # print('lin combo=' + str(lin_combo) + ' r=' + str(radius))
    # plot_multi_guessed_paths(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,np.linspace(0,sampleingtime[-1],numpoints))
    return guessed_paths


def plot_multi_sim_path(sim_paths,beta,gamma,epsilon_matrix,list_sims,tf):
    lam = beta / gamma
    x0,s0 = (lam - 1) / lam, 1 / lam + np.log(lam) - 1
    A_w,A_u,alpha_list_w,alpha_list_u,path_list=[],[],[],[],[]

    # f = open('shooting_path_lam_' + str(lam).replace('.', '') + '.csv', "w+")
    # with f:
    #     writer = csv.writer(f)
    #     writer.writerows([Epsilon, fluctuation, Num_meauserments, T_final])

    count=0
    for guessed_paths,case_to_run,list_of_epsilons in zip(sim_paths,list_sims,epsilon_matrix):

        sim_label_x= 'const_eps_lam' if count==0 else 'const_eps_mu'
        sim_label= 'const eps mu' if case_to_run is 'la' else 'const eps lam' if case_to_run is 'al' else sim_label_x
        path_list.append(guessed_paths)
        save_label=sim_label_x if case_to_run is 'x' else case_to_run
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            # epsilon_numerical= epsilon_mu if case_to_run is 'la' else epsilon_lam
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam


            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu

            pw_for_path = path[:, 2] + path[:, 3]
            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            # pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_mu ** 2) * epsilon_mu) if case_to_run is 'la' else -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * epsilon_lam)
            pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (1 + ((1 + alpha) / lam) * epsilon_mu ** 2) * lam) if case_to_run is 'la' else -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * lam)
            plt.plot(w_for_path_clancy, (pw_for_path - pw0_clancy) / epsilon_numerical ** 2, linewidth=4,
                     label='Numerical eps=' + str(epsilon))

            pw_thoery= -2*np.log(lam-2*w_for_path_clancy*lam) + ((((4* w_for_path_clancy * (1 + alpha)*lam)/(1 - 2* w_for_path_clancy) - alpha* (1 + (-1 + 2* w_for_path_clancy)* lam) * (alpha+ (2 +alpha)*lam)))/lam**2) *epsilon_numerical**2 if case_to_run is 'la' else -2*np.log(lam-2*w_for_path_clancy*lam) -((((2 *2*w_for_path_clancy *alpha* (1 + alpha) *lam)/(-1 +
     2*w_for_path_clancy) + (1 + (-1 + 2*w_for_path_clancy) *lam) * (1 +lam +2 *alpha* lam)))/lam**2)*epsilon_numerical**2
            if case_to_run is not 'x':
                plt.plot(w_for_path_clancy,(pw_thoery-pw0_clancy)/epsilon_numerical**2,linewidth=4,label='Theory eps=' + str(epsilon),linestyle=':')
        plt.xlabel('w')
        plt.ylabel('(pw-pw0)/eps^2')
        plt.title('(pw-pw0)/eps^2 vs w, lam=' + str(lam)+' '+sim_label)
        plt.legend()
        plt.savefig('pw_vs_w_'+save_label + '.png', dpi=500)
        plt.show()

        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            pw_for_path = path[:, 2] + path[:, 3]
            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            plt.plot(w_for_path_clancy,pw_for_path, linewidth=4,label='Numerical eps=' + str(epsilon))
        plt.xlabel('w')
        plt.ylabel('pw')
        plt.title('pw vs w, lam=' + str(lam) + ' ' + sim_label)
        plt.legend()
        plt.savefig('pw_vs_w_non_norm_' + save_label + '.png', dpi=500)
        plt.show()

        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu

            y1=path[:, 0]
            p1=path[:, 2]
            px1_0= -np.log(lam*(1-2*y1*(1+(1/lam)*epsilon_mu+((1+lam+alpha*lam)/(lam**2))*epsilon_mu**2))) if case_to_run is 'la' else -np.log(lam*(1-2*y1*(1+(alpha/lam)*epsilon_lam+(alpha*(alpha+lam+alpha*lam)/lam**2)*epsilon_lam**2)))
            plt.plot(y1, (p1-px1_0), linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('x1')
        plt.ylabel('p1-p1_0')
        plt.title('p1 vs x1 normalized, lam=' + str(lam) + ' ' + sim_label)
        plt.legend()
        plt.savefig('p1_v_x1_norm' + save_label + '.png', dpi=500)
        plt.show()


        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu

            y2=path[:, 1]
            p2=path[:, 3]
            px2_0= -np.log(lam*(1-2*y2*(1-(1/lam)*epsilon_mu+((1+lam+alpha*lam)/(lam**2))*epsilon_mu**2))) if case_to_run is 'la' else -np.log(lam*(1-2*y2*(1-(alpha/lam)*epsilon_lam+(alpha*(alpha+lam+alpha*lam)/lam**2)*epsilon_lam**2)))
            plt.plot(y2, (p2-px2_0), linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('x2')
        plt.ylabel('p2-p2_0')
        plt.title('p2 vs x2 normalized, lam=' + str(lam) + ' ' + sim_label)
        plt.legend()
        plt.savefig('p2_v_x2_norm' + save_label + '.png', dpi=500)
        plt.show()

        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu

            pw_for_path = path[:, 2] + path[:, 3]
            u_for_path = (path[:, 0] - path[:, 1]) / 2
            # pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_mu ** 2) * epsilon_mu) if case_to_run is 'la' else -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * epsilon_lam)
            pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (1 + ((1 + alpha) / lam) * epsilon_mu ** 2) * lam) if case_to_run is 'la' else -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * lam)
            plt.plot(u_for_path/epsilon_numerical, (pw_for_path - pw0_clancy) / epsilon_numerical ** 2, linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('u/eps')
        plt.ylabel('(pw-pw0)/eps^2')
        plt.title('((pw-pw0)/eps^2 vs u/eps, lam=' + str(lam)+' '+sim_label)
        plt.legend()
        plt.savefig('pw_vs_u_'+save_label + '.png', dpi=500)
        plt.show()


        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam
            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            u_for_path = (path[:, 0] - path[:, 1]) / 2
            plt.plot(w_for_path_clancy, u_for_path / epsilon_numerical, linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('w')
        plt.ylabel('u/eps')
        plt.title('u/eps vs w, lam=' + str(lam)+' '+sim_label)
        plt.legend()
        plt.savefig('w_v_u_numerical_' + save_label + '.png', dpi=500)
        plt.show()


        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            pu_for_path = path[:, 2] - path[:, 3]
            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            plt.plot(w_for_path_clancy, pu_for_path / epsilon_numerical, linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('w')
        plt.ylabel('pu/eps')
        plt.title('pu/eps vs w, lam=' + str(lam)+ ' '+sim_label)
        plt.legend()
        plt.savefig('pu_v_w_numerical_' + save_label + '.png', dpi=500)
        plt.show()


        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu


            pu_for_path = path[:, 2] - path[:, 3]
            u_for_path = (path[:, 0] - path[:, 1]) / 2
            plt.plot(u_for_path / epsilon_numerical, pu_for_path / epsilon_numerical, linewidth=4,
                     label='Numerical eps=' + str(epsilon))
            pu_theory_alpha = 2 * x0 * epsilon_lam + (4 * lam * u_for_path) / alpha if case_to_run is 'al' else 2*alpha*x0*epsilon_mu+4*lam*alpha*u_for_path
            if case_to_run is not 'x':
                plt.plot(u_for_path / epsilon_numerical, pu_theory_alpha / epsilon_numerical, linewidth=4,
                         label='Theory eps=' + str(epsilon),linestyle=':')
        plt.xlabel('u/eps')
        plt.ylabel('pu/eps')
        plt.title('pu/eps vs u, lam=' + str(lam)+ ' '+ sim_label)
        plt.legend()
        plt.savefig('pu_v_u_numerical_' + save_label + '.png', dpi=500)
        plt.show()


        A_numerical, A_theory, alpha_list_w, A_numerical_norm = [], [], [], []
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu


            # alpha = epsilon_mu / epsilon_lam if 'la' else epsilon_lam / epsilon_mu
            alpha_list_w.append(alpha)
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, \
            dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf, gamma)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]

            y1_for_linear = np.linspace(path[:, 0][-1], 0, 1000)
            py1_linear = p1_star_clancy - ((p1_star_clancy - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
            y2_for_linear = np.linspace(path[:, 1][-1], 0, 1000)
            py2_linear = p2_star_clancy - ((p2_star_clancy - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
            I_addition_to_path = simps(py1_linear + py2_linear, (y1_for_linear + y2_for_linear) / 2)

            integral_numeric = simps(path[:, 2] + path[:, 3], (path[:, 0] + path[:, 1]) / 2)
            integral_numeric_correction = integral_numeric + I_addition_to_path - s0
            A_numerical.append(integral_numeric_correction)
            A_numerical_norm.append(integral_numeric_correction / epsilon_numerical ** 2)
        A_w.append(A_numerical_norm)
        plt.plot(alpha_list_w, A_numerical_norm, linewidth=4, linestyle='None', Marker='o', label='Numerical',
                 markersize=10)
        alpha_list_for_theory=np.linspace(alpha_list_w[0],alpha_list_w[-1],1000)
        theory_I_w = np.array(
            [  (-((lam-1)*(-1+lam*(lam+2*a*(-3-2*a+lam))))/(4*lam**3)-a*(1+a)*np.log(lam)/lam) for a in
             alpha_list_for_theory]) if case_to_run is 'al' else np.array(
            [(-((-1 + lam)*(-4*lam +2*a*(-3 +lam)*lam +a**2*(-1 + lam**2)))/(4*lam**3) -((1 + a)*np.log(lam))/lam) for a in
             alpha_list_for_theory])
        if case_to_run is not 'x':
            plt.plot(alpha_list_for_theory, theory_I_w, linewidth=4, linestyle=':', label='Theory')
        plt.xlabel('alpha')
        plt.ylabel('Iw/eps^2')
        plt.title('Iw/eps^2 vs alpha, lam=' + str(lam)+ ' '+sim_label)
        plt.legend()
        plt.savefig('Iw_v_eps_alpha_'+save_label + '.png', dpi=500)
        plt.show()



        A_numerical, A_theory, alpha_list_w, A_numerical_norm = [], [], [], []
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu


            alpha_list_w.append(alpha)
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, \
            dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf, gamma)

            y1_for_linear = np.linspace(path[:, 0][-1], 0, 1000)
            py1_linear = p1_star_clancy - ((p1_star_clancy - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
            y2_for_linear = np.linspace(path[:, 1][-1], 0, 1000)
            py2_linear = p2_star_clancy - ((p2_star_clancy - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
            I_addition_to_path = simps(py1_linear, y1_for_linear)

            integral_numeric = simps(path[:, 2],path[:, 0])
            integral_numeric_correction = integral_numeric + I_addition_to_path - s0
            A_numerical.append(integral_numeric_correction)
            A_numerical_norm.append(integral_numeric_correction)
        A_w.append(A_numerical_norm)
        plt.plot(alpha_list_w, A_numerical_norm, linewidth=4, linestyle='None', Marker='o', label='Numerical',
                 markersize=10)
        plt.xlabel('alpha')
        plt.ylabel('Ix1')
        plt.title('Ix1 vs alpha, lam=' + str(lam)+ ' '+sim_label)
        plt.legend()
        plt.savefig('Ix1_v_eps_alpha_'+save_label + '.png', dpi=500)
        plt.show()





        A_numerical, A_theory, alpha_list_w, A_numerical_norm = [], [], [], []
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu


            alpha_list_w.append(alpha)
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, \
            dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf, gamma)

            y2_for_linear = np.linspace(path[:, 1][-1], 0, 1000)
            py2_linear = p2_star_clancy - ((p2_star_clancy - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
            I_addition_to_path = simps(py2_linear, y2_for_linear)

            integral_numeric = simps(path[:, 3],path[:, 1])
            integral_numeric_correction = integral_numeric + I_addition_to_path - s0
            A_numerical.append(integral_numeric_correction)
            A_numerical_norm.append(integral_numeric_correction)
        A_w.append(A_numerical_norm)
        plt.plot(alpha_list_w, A_numerical_norm, linewidth=4, linestyle='None', Marker='o', label='Numerical',
                 markersize=10)
        plt.xlabel('alpha')
        plt.ylabel('Ix2')
        plt.title('Ix2 vs alpha, lam=' + str(lam)+ ' '+sim_label)
        plt.legend()
        plt.savefig('Ix2_v_eps_alpha_'+save_label + '.png', dpi=500)
        plt.show()





        A_numerical,A_theory,alpha_list_u,A_numerical_norm=[],[],[],[]
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]

            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam


            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu

            alpha_list_u.append(alpha)
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy,\
            dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf,gamma)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]

            y1_for_linear=np.linspace(path[:,0][-1],0,1000)
            py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
            y2_for_linear=np.linspace(path[:,1][-1],0,1000)
            py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear

            I_addition_to_path=simps(py1_linear-py2_linear,(y1_for_linear-y2_for_linear)/2)
            pudu = simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1]) / 2))
            # A_numerical_norm.append((pudu+I_addition_to_path)/epsilon_numerical**2)
            A_numerical_norm.append((pudu+I_addition_to_path)/epsilon_numerical**2)
        A_u.append(A_numerical_norm)
        plt.plot(alpha_list_u,A_numerical_norm,linewidth=4,linestyle='None', Marker='o', label='Numerical',markersize=10)

        alpha_list_for_theory=np.linspace(alpha_list_u[0],alpha_list_u[-1],1000)
        theory_I_u = [a*((lam-1)**2)/(2*lam**3) for a in alpha_list_for_theory]
        if case_to_run is not 'x':
            plt.plot(alpha_list_for_theory, theory_I_u, linewidth=4, linestyle=':', label='Theory')
        plt.xlabel('alpha')
        plt.ylabel('Iu/eps^2')
        plt.title('Iu/eps^2 vs alpha lam='+str(lam)+ ' ' +  sim_label)
        plt.legend()
        plt.tight_layout()
        plt.savefig('pudu_v_eps_'+save_label + '.png', dpi=500)
        plt.show()

        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical = epsilon_mu if case_to_run is 'la' else epsilon_lam
            plt.plot(path[:, 0], path[:, 2], linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('x1')
        plt.ylabel('p1')
        plt.title('p1 vs x1, lam=' + str(lam) + ' ' + sim_label)
        plt.legend()
        plt.savefig('p1_v_x1_' + save_label + '.png', dpi=500)
        plt.show()

        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical = epsilon_mu if case_to_run is 'la' else epsilon_lam
            plt.plot(path[:, 1], path[:, 3], linewidth=4,
                     label='Numerical eps=' + str(epsilon))
        plt.xlabel('x2')
        plt.ylabel('p2')
        plt.title('p2 vs x2, lam=' + str(lam) + ' ' + sim_label)
        plt.legend()
        plt.savefig('p2_v_x2_' + save_label + '.png', dpi=500)
        plt.show()
        count=count+1


    for action_w,action_u,case_to_run in zip(A_w,A_u,list_sims):
        s1=[iu+iw for iw,iu in zip(action_w,action_u)]
        plt.plot(alpha_list_w,s1,linewidth=4,label='Numerical ' + case_to_run,linestyle='None', Marker='o',markersize=10)
        alpha_list_for_theory=np.linspace(alpha_list_u[0],alpha_list_u[-1],1000)
        theory_I_w = np.array(
            [  (-((lam-1)*(-1+lam*(lam+2*a*(-3-2*a+lam))))/(4*lam**3)-a*(1+a)*np.log(lam)/lam) for a in
             alpha_list_for_theory]) if case_to_run is 'al' else np.array(
            [(-((-1 + lam)*(-4*lam +2*a*(-3 +lam)*lam +a**2*(-1 + lam**2)))/(4*lam**3) -((1 + a)*np.log(lam))/lam) for a in
             alpha_list_for_theory])
        theory_I_u =np.array( [a*((lam-1)**2)/(2*lam**3) for a in alpha_list_for_theory])
        s_theory=theory_I_u+theory_I_w
        if case_to_run is not 'x':
            plt.plot(alpha_list_for_theory,s_theory,linewidth=4,label='Theory ' + case_to_run,linestyle='--')
    plt.xlabel('alpha')
    plt.ylabel('s1')
    plt.title('s1 vs alpha lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('s1_la_and_al' + '.png', dpi=500)
    plt.show()

    for action_w,action_u,case_to_run in zip(A_w,A_u,list_sims):
        ratio=[np.absolute(iw)/np.absolute(iu) for iw,iu in zip(action_w,action_u)]
        plt.plot(alpha_list_w,ratio,linewidth=4,label='Numerical ' + case_to_run,linestyle='None', Marker='o',markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('ratio')
    plt.title('Ratio vs alpha lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('ratio_iw_iu' + '.png', dpi=500)
    plt.show()

    #In the same fig integrals

    for action_w,case_to_run in zip(A_w,list_sims):
        plt.plot(alpha_list_w,action_w,linewidth=4,label='Numerical ',linestyle='None', Marker='o',markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('Iw')
    plt.title('Iw vs alpha lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('Iw_togther' + '.png', dpi=500)
    plt.show()

    for action_u,case_to_run in zip(A_u,list_sims):
        plt.plot(alpha_list_w,action_u,linewidth=4,label='Numerical ',linestyle='None', Marker='o',markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('Iu')
    plt.title('Iu vs alpah lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('Iu_togther' + '.png', dpi=500)
    plt.show()

    lines = ["-", "--"]
    linecycler = cycle(lines)

    for guessed_paths, case_to_run, list_of_epsilons in zip(sim_paths, list_sims, epsilon_matrix):
        line_style = next(linecycler)
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu


            pw_for_path = path[:, 2] + path[:, 3]
            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (1 + ((1 + alpha) / lam) * epsilon_mu ** 2) * lam) if case_to_run is 'la' else -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * lam)
            plt.plot(w_for_path_clancy, (pw_for_path - pw0_clancy) / epsilon_numerical ** 2, linewidth=4,
                     label='eps=' + str(epsilon)+' case='+str(case_to_run),linestyle=line_style)
    plt.xlabel('w')
    plt.ylabel('(pw-pw0)/eps^2')
    plt.title('(pw-pw0)/eps^2 vs w lam=' + str(lam) )
    plt.legend()
    plt.tight_layout()
    plt.savefig('pw_v_w_togther' + '.png', dpi=500)
    plt.show()


    for guessed_paths, case_to_run, list_of_epsilons in zip(sim_paths, list_sims, epsilon_matrix):
        line_style = next(linecycler)
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            if epsilon_mu==0.0 or epsilon_lam==0.0:
                alpha=0
            elif case_to_run is 'la':
                alpha=epsilon_lam/epsilon_mu
            elif case_to_run is 'al':
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==0:
                alpha=epsilon_mu / epsilon_lam
            elif case_to_run is 'x' and count==1:
                alpha=epsilon_lam/epsilon_mu


            pw_for_path = path[:, 2] + path[:, 3]
            u_for_path= (path[:, 0] - path[:, 1]) / 2
            pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (1 + ((1 + alpha) / lam) * epsilon_mu ** 2) * lam) if case_to_run is 'la' else -2 * np.log(lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * lam)
            plt.plot(u_for_path/epsilon_numerical, (pw_for_path - pw0_clancy) / epsilon_numerical ** 2, linewidth=4,
                     label='eps=' + str(epsilon)+' case='+str(case_to_run),linestyle=line_style)
    plt.xlabel('u/eps')
    plt.ylabel('(pw-pw0)/eps^2')
    plt.title('(pw-pw0)/eps^2 vs u lam=' + str(lam) )
    # plt.legend()
    plt.tight_layout()
    plt.savefig('pw_v_u_togther' + '.png', dpi=500)
    plt.show()


    lines = ["-", "--"]
    linecycler = cycle(lines)

    for guessed_paths, case_to_run, list_of_epsilons in zip(sim_paths, list_sims, epsilon_matrix):
        line_style=next(linecycler)
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            pu_for_path = path[:, 2] - path[:, 3]
            u_for_path= (path[:, 0] - path[:, 1]) / 2
            plt.plot(u_for_path/epsilon_numerical, pu_for_path/epsilon_numerical, linewidth=4,
                     label='eps=' + str(epsilon)+' case='+sim_label,linestyle=line_style)
    plt.xlabel('u/eps')
    plt.ylabel('pu/eps')
    plt.title('pu/eps vs_u/eps lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('pu_v_u_togther' + '.png', dpi=500)
    plt.show()


    for guessed_paths, case_to_run, list_of_epsilons in zip(sim_paths, list_sims, epsilon_matrix):
        line_style=next(linecycler)
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            u_for_path= (path[:, 0] - path[:, 1]) / 2
            plt.plot(w_for_path_clancy, u_for_path / epsilon_numerical, linewidth=4,
                     label='eps=' + str(epsilon)+' case='+str(case_to_run),linestyle=line_style)
    plt.xlabel('w')
    plt.ylabel('u/eps')
    plt.title('w vs u/eps lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('w_v_u_togther' + '.png', dpi=500)
    plt.show()

    for guessed_paths, case_to_run, list_of_epsilons in zip(sim_paths, list_sims, epsilon_matrix):
        line_style=next(linecycler)
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical = epsilon_mu if case_to_run is 'la' else epsilon_lam

            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            pu_for_path = path[:, 2] - path[:, 3]
            plt.plot(w_for_path_clancy, pu_for_path/epsilon_numerical, linewidth=4,
                     label='eps=' + str(epsilon)+' case='+str(case_to_run),linestyle=line_style)
    plt.xlabel('w')
    plt.ylabel('pu/eps')
    plt.title('w vs pu/eps lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('w_v_pu_togther' + '.png', dpi=500)
    plt.show()

    for guessed_paths, case_to_run, list_of_epsilons in zip(sim_paths, list_sims, epsilon_matrix):
        line_style=next(linecycler)
        for path, epsilon in zip(guessed_paths, list_of_epsilons):
            epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
            epsilon_numerical= epsilon_mu if case_to_run is 'la' or (case_to_run is 'x' and count==1) else epsilon_lam

            pu_for_path = path[:, 2] - path[:, 3]
            w_for_path_clancy = (path[:, 0] + path[:, 1]) / 2
            plt.plot(w_for_path_clancy, pu_for_path/epsilon_numerical, linewidth=4,
                     label='eps=' + str(epsilon)+' case='+sim_label,linestyle=line_style)
    plt.xlabel('w')
    plt.ylabel('pu/eps')
    plt.title('pu/eps vsu/eps lam=' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('pu_v_w_togther' + '.png', dpi=500)
    plt.show()


p1_clancy_epsmu = lambda path,lam,epsilon:np.array([np.log(y1/(lam*(1-epsilon)*(1/2-y1)*(y1+y2))) for y1,y2 in zip(path[:,0],path[:,1])])
p2_clancy_epsmu = lambda path,lam,epsilon:np.array([np.log(y2/(lam*(1+epsilon)*(1/2-y2)*(y1+y2)))for y1,y2 in zip(path[:,0],path[:,1])])


path_sub_theory = lambda path_numerical,lam,epsilon:np.array([path_numerical[:,0],path_numerical[:,1],path_numerical[:,2]-p1_clancy_epsmu(path_numerical,lam,epsilon),path_numerical[:,3]-p2_clancy_epsmu(path_numerical,lam,epsilon)]).T

d_eps_mu= lambda lam,eps:-(-2+lam-lam*eps**2+np.sqrt(lam**2-2*(-2+lam**2)*eps**2+lam**2*eps**4))/(2*(-1+eps**2))
A_sus_alone_clancy= lambda lam,eps:(1/2)*(np.log(1+(1-eps)*d_eps_mu(lam,eps)) +np.log(1+(1+eps)*d_eps_mu(lam,eps)))-(1/lam)*d_eps_mu(lam,eps)

def plot_eps_mu_sub(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,tf):
    lam=beta/gamma
    x0,s0=(lam-1)/lam,1/lam+np.log(lam)-1
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        path_sub=path_sub_theory(path,lam,epsilon_mu)
        pu_sub=path_sub[:,2]-path_sub[:,3]
        u_for_path=(path[:,0]-path[:,1])/2
        integral_numeric=-(simps((path_sub[:, 2] - path_sub[:, 3]), ((path_sub[:, 0] - path_sub[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(u_for_path/epsilon_mu,pu_sub/epsilon_mu,linewidth=4,label='Numerical eps='+str(epsilon))
        # plt.plot(pu_for_path/epsilon_lam,u_theory_full/epsilon_lam,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('u/eps')
    plt.ylabel('pu/eps')
    plt.title('pu/eps vs u/eps sub, lam='+str(lam))
    plt.legend()
    plt.savefig('pu_v_u_sub_epsmu_const'+'.png',dpi=500)
    plt.show()

    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        path_sub=path_sub_theory(path,lam,epsilon_mu)
        pw_sub=path_sub[:,2]+path_sub[:,3]
        w_for_path=(path_sub[:,0]+path_sub[:,1])/2
        integral_numeric=-(simps((path_sub[:, 2] - path_sub[:, 3]), ((path_sub[:, 0] - path_sub[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(w_for_path,pw_sub,linewidth=4,label='Numerical eps='+str(epsilon))
    plt.xlabel('w')
    plt.ylabel('pw')
    plt.title('pw vs w sub, lam='+str(lam))
    plt.legend()
    plt.savefig('pw_v_w_sub_epsmu_const'+'.png',dpi=500)
    plt.show()

    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        path_sub=path_sub_theory(path,lam,epsilon_mu)
        integral_numeric=-(simps((path_sub[:, 2] - path[:, 3]), ((path_sub[:, 0] - path_sub[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(path_sub[:,0],path_sub[:,2],linewidth=4,label='Numerical eps='+str(epsilon))
        # plt.plot(pu_for_path/epsilon_lam,u_theory_full/epsilon_lam,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('x1')
    plt.ylabel('p1')
    plt.title('p1 vs x1 sub, lam='+str(lam))
    plt.legend()
    plt.savefig('p1_v_x1_sub_epsmu_const'+'.png',dpi=500)
    plt.show()

    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        path_sub=path_sub_theory(path,lam,epsilon_mu)
        integral_numeric=-(simps((path_sub[:, 2] - path[:, 3]), ((path_sub[:, 0] - path_sub[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(path_sub[:,1],path_sub[:,3],linewidth=4,label='Numerical eps='+str(epsilon))
    plt.xlabel('x2')
    plt.ylabel('p2')
    plt.title('p2 vs x2 sub, lam='+str(lam))
    plt.legend()
    plt.savefig('p2_v_x2_sub_epsmu_const'+'.png',dpi=500)
    plt.show()

    integration_list,alpha_list=[],[]
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        path_sub=path_sub_theory(path,lam,epsilon_mu)
        integration_list.append(simps(path_sub[:, 2], path_sub[:, 0])+simps(path[:,3],path[:,1]))
        alpha_list.append(epsilon_lam/epsilon_mu)
    plot(alpha_list,(np.array(integration_list))/epsilon_mu**2,linewidth=4,label='Sim',linestyle='None', Marker='o', markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('S1')
    plt.title('S1/epsilon_mu^2 vs alpha sub, lam='+str(lam))
    plt.legend()
    plt.savefig('s1_v_alpha_sub_epsmu_const'+'.png',dpi=500)
    plt.show()


    integration_list,alpha_list=[],[]
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        integration_list.append(simps(path[:, 2], path[:, 0])+simps(path[:,3],path[:,1])-A_sus_alone_clancy(lam,epsilon_mu))
        alpha_list.append(epsilon_lam/epsilon_mu)
    plot(alpha_list,(np.array(integration_list))/epsilon_mu**2,linewidth=4,label='Sim',linestyle='None', Marker='o', markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('S1')
    plt.title('S1/epsilon_mu^2 vs alpha sub, lam='+str(lam))
    plt.legend()
    plt.savefig('s1_v_alpha_minus_clancy_epsmu_const'+'.png',dpi=500)
    plt.show()



def plot_eps_lam_sub(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,tf):
    lam=beta/gamma
    x0=(lam-1)/lam

    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        w_for_path, u_for_path = (path[:, 0] + path[:, 1]) / 2, (path[:, 0] - path[:, 1]) / 2
        pu_theory = np.array([-np.log(1 + (1 - epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) + np.log(
            1 + (1 + epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) for w, u in
                              zip(w_for_path, u_for_path)])
        pu_numerical=path[:,2]-path[:,3]
        plt.plot(u_for_path/epsilon_lam, (pu_numerical-pu_theory)/epsilon_lam, linestyle='--', linewidth=4,label='eps='+str(epsilon))
    plt.xlabel('u')
    plt.ylabel('pu')
    plt.title('pu vs u sub, lam='+str(lam))
    plt.legend()
    plt.savefig('pu_v_u_sub_epslam_const'+'.png',dpi=500)
    plt.show()

    alpha_list,action_list=[],[]
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, J = \
            eq_hamilton_J(case_to_run, beta, epsilon, t,gamma)
        q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
        f_of_d = (1 / 2) * (beta / gamma) * (1 - epsilon_lam ** 2)
        D = (-1 + f_of_d + np.sqrt(epsilon_lam ** 2 + f_of_d ** 2)) / (1 - epsilon_lam ** 2)
        A_theory = -(1 / 2) * (q_star[2] + q_star[3]) - (gamma / beta) * D
        action_list.append(simps(path[:, 2], path[:, 0])+simps(path[:,3],path[:,1])-A_theory)
        alpha_list.append(epsilon_mu/epsilon_lam)
    plt.plot(alpha_list, np.array(action_list)/epsilon_mu**2, linestyle='None', Marker='o', markersize=10,label='sim')
    plt.xlabel('1/alpha')
    plt.ylabel('S1/eps_mu^2')
    plt.title('s1/eps_mu^2 vs 1/alpha sub, lam='+str(lam))
    plt.legend()
    plt.savefig('s1_v_1overalpha_epslamconst'+'.png',dpi=500)
    plt.show()


def plot_eps_mu_clancy(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,tf):
    lam=beta/gamma
    x0=(lam-1)/lam
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        p1_theory=p1_clancy_epsmu(path,lam,epsilon_mu)
        p2_theory=p2_clancy_epsmu(path,lam,epsilon_mu)
        pu_theory=p1_theory-p2_theory
        pu_for_path=path[:,2]-path[:,3]
        u_for_path=(path[:,0]-path[:,1])/2
        integral_numeric=-(simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(u_for_path/epsilon_mu,pu_for_path/epsilon_mu,linewidth=4,label='Numerical eps='+str(epsilon))
        plt.plot(u_for_path/epsilon_mu,pu_theory/epsilon_mu,linewidth=4,label='Theory eps='+str(epsilon),linestyle='--')
        # plt.plot(pu_for_path/epsilon_lam,u_theory_full/epsilon_lam,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('u/eps')
    plt.ylabel('pu/eps')
    plt.title('pu/eps vs u/eps, lam='+str(lam))
    plt.legend()
    plt.savefig('pu_v_u_epsmu_const'+'.png',dpi=500)
    plt.show()

    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        p1_theory=p1_clancy_epsmu(path,lam,epsilon_mu)
        integral_numeric=-(simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(path[:,0],path[:,2],linewidth=4,label='Numerical eps='+str(epsilon))
        plt.plot(path[:,0],p1_theory,linewidth=4,label='Theory eps='+str(epsilon),linestyle='--')
        # plt.plot(pu_for_path/epsilon_lam,u_theory_full/epsilon_lam,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('x1')
    plt.ylabel('p1')
    plt.title('p1 vs x1, lam='+str(lam))
    plt.legend()
    plt.savefig('p1_v_x1_epsmu_const'+'.png',dpi=500)
    plt.show()

    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        p2_theory=p2_clancy_epsmu(path,lam,epsilon_mu)
        integral_numeric=-(simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1] )/2)))
        # integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(path[:,1],path[:,3],linewidth=4,label='Numerical eps='+str(epsilon))
        plt.plot(path[:,1],p2_theory,linewidth=4,label='Theory eps='+str(epsilon),linestyle='--')
        # plt.plot(pu_for_path/epsilon_lam,u_theory_full/epsilon_lam,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('x2')
    plt.ylabel('p2')
    plt.title('p2 vs x2, lam='+str(lam))
    plt.legend()
    plt.savefig('p2_v_x2_epsmu_const'+'.png',dpi=500)
    plt.show()


def plot_multi_guessed_paths(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,tf):
    if type(beta) is not list:
        lam=beta/gamma
        x0=(lam - 1) / lam
        epsilon_mu_changes =  False
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:(p[:, 0] - p[:, 1]) / 2,lambda p,eps,x:p[:, 2] - p[:, 3]
#                  ,'zw','u','pu',x0,lam,'p_u vs u','pu_vs_u',case_to_run,tf)
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] - p[:, 1]) / 2)/(eps*alph),lambda p,eps,x:(p[:, 2] - p[:, 3]-2*x*eps)/eps
#                  ,'au','u/(alpha*eps)','pu/eps-2x0',x0,lam,'pu/eps-2x0 vs u/(eps*alpha)','pu_vs_u_norm',case_to_run,tf)
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] + p[:, 1]) / 2),lambda p,eps,x:p[:, 2] + p[:, 3] - 2 * np.log(gamma / (beta * (1 - (p[:, 0] + p[:, 1])
# )))
#                  ,'wpw','w','pw-pw0',x0,lam,'pw vs w','pw_vs_w',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] + p[:, 1]) / 2),lambda p,eps,x:(p[:, 2] - p[:, 3] )/eps
#                  ,'wpu','w','pu',x0,lam,'pu vs w','second_order_pu_w',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] - p[:, 1]) / 2),
#     lambda p,eps,x:p[:, 2] + p[:, 3] -2 * np.log(gamma / (beta * (1 - (p[:,0] + p[:,1])/(1+eps**2)))),
#     'upw0','u','(pw-pw0)',x0,lam,'(pw-pw0) vs u','pw_v_u_miki',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] - p[:, 1]) / 2)/eps,
#     lambda p,eps,x:(p[:, 2] + p[:, 3] -2 * np.log(gamma / (beta * (1 - (p[:,0] + p[:,1])/(1+eps**2)))))/eps**2,
#     'upw0n','u','(pw-pw0)/eps^2',x0,lam,'Normalized (pw-pw0)/eps^2 vs u/eps','pw_v_u_norm_miki',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] + p[:, 1]) / 2),
#     lambda p,eps,x:(p[:, 2] + p[:, 3] -2 * np.log(gamma / (beta * (1 - (p[:,0] + p[:,1])))))/eps**2,
#     'wpwl','w','(pw-pw0)/eps^2',x0,lam,'(pw-pw0)/eps^2 vs w','pw_vs_w_norm',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] - p[:, 1]) / 2)/(eps*alph),
#     lambda p,eps,x:(p[:, 2] - p[:, 3] - 2 * x* eps)/eps,
#     'uvpua','u/(alpha*eps)','pu/eps-2x0',x0,lam,'pu/eps-2x0*eps vs u/(eps*alpha) vs w','u_norm_v_pu_minus_2x0',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:((p[:, 0] + p[:, 1]) / 2),
#     lambda p,eps,x:((p[:, 0] - p[:, 1] )/2)/eps,
#     'wvu','w','u/eps',x0,lam,'U vs W','u_v_w',case_to_run,tf)
#
#     generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:(p[:, 2] - p[:, 3])/eps,
#     lambda p,eps,x:((p[:, 0] - p[:, 1] )/2)/eps,
#     'puiu','pu','u/eps',x0,lam,'pu/eps vs u/eps','u_v_pu_norm',case_to_run,tf,lambda p,eps: ' I='+str(round(simps((p[:, 2] - p[:, 3])/eps, ((p[:, 0] - p[:, 1] )/2)/eps),4)))


#     if type(beta) is not list:
#         A_aprox_theory,A_exact_theory,A_numerical=[],[],[]
#         s0=1/lam+np.log(lam)-1
#         for path, epsilon in zip(guessed_paths, list_of_epsilons):
#             epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
#             y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy,\
#             dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf,gamma)
#             q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
#
#             y1_for_linear=np.linspace(path[:,0][-1],0,1000)
#             py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
#             y2_for_linear=np.linspace(path[:,1][-1],0,1000)
#             py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear
#             addition_to_path = np.stack((y1_for_linear,y2_for_linear,py1_linear,py2_linear),axis=1)
#
#             I_addition_to_path=simps(py1_linear,y1_for_linear)+simps(py2_linear,y2_for_linear)
#
#             path_addition=np.vstack((path,addition_to_path))
#
#             f_of_d = (1 / 2) * (beta / gamma) * (1 - epsilon_lam ** 2)
#             D = (-1 + f_of_d + np.sqrt(epsilon_lam ** 2 + f_of_d ** 2)) / (1 - epsilon_lam ** 2)
#             I_pu_aprox=(((lam - 1) ** 3) / (4 * lam ** 3))*epsilon_lam**2
#             I_pw_aprox=-((lam+1)*(lam-1)**2/(4*lam**3))*epsilon_lam**2
#             A_aprox_theory.append(-I_pu_aprox+I_pw_aprox)
#             A_exact_theory_full=-(1 / 2) * (q_star[2] + q_star[3]) - (gamma / beta) * D
#             A_exact_theory.append(A_exact_theory_full-s0)
#
#             # pudu=simps((path_new[:, 2] - path_new[:, 3]), ((path_new[:, 0] - path_new[:, 1] )/2))
#             # pwdw=simps((path_new[:, 2] + path_new[:, 3]), ((path_new[:, 0] + path_new[:, 1] )/2))
#             # pwdwcor=pwdw-s0
#             # A_numerical.append(pwdwcor+pudu)
#
#             # pudu_org=simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1] )/2))
#             # pwdw_org=simps((path[:, 2] + path[:, 3]), ((path[:, 0] + path[:, 1] )/2))
#             # pwdwcor_org=pwdw-s0
#             # A_numerical_org=pwdwcor_org+pudu_org
#
#
#             # pudu_add=simps((path_addition[:, 2] - path_addition[:, 3]), ((path_addition[:, 0] - path_addition[:, 1] )/2))
#             # pwdw_add=simps((path_addition[:, 2] + path_addition[:, 3]), ((path_addition[:, 0] + path_addition[:, 1] )/2))
#             # py1_add=simps(path_addition[:, 2], path_addition[:, 0])
#             # py2_add=simps((path_addition[:, 3], path_addition[:, 1]))
#
#
#             # pwdwcor_add=pwdw-s0
#             # A_numerical_add=pwdwcor_add+pudu_add
#
#             # generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:(p[:, 0] - p[:, 1]) / 2,lambda p,eps,x:p[:, 2] - p[:, 3]
#             #              ,'zw','u','pu',x0,lam,'p_u vs u','pu_vs_u',case_to_run,tf)
#
#             guessed_paths=[path_addition]
#             # generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:(p[:, 0] - p[:, 1]) / 2,lambda p,eps,x:p[:, 2] - p[:, 3]
#             #              ,'zw','u','pu',x0,lam,'p_u vs u','pu_vs_u_after',case_to_run,tf)
#
#             A_integration_y = simps(path[:, 2], path[:, 0]) + simps(path[:, 3], path[:, 1])-s0+I_addition_to_path
#             A_numerical.append(A_integration_y)
#         # fig = plt.figure()
#         # ax = fig.add_subplot(1, 1, 1)
#         plt.plot(np.array(list_of_epsilons)[:,0]**2,A_numerical,linewidth=4,linestyle='None', Marker='o', label='Numerical',markersize=10)
#         plt.plot(np.array(list_of_epsilons)[:,0]**2,A_exact_theory,linewidth=4,linestyle='None', Marker='^', label='Clancy',markersize=10)
#         plt.plot(np.array(list_of_epsilons)[:,0]**2,A_exact_theory,linewidth=4,linestyle='None', Marker='X', label='Miki',markersize=10)
#         plt.plot(np.array(list_of_epsilons)[:,0]**2,-(lam-1)**2/(2*lam**2)*np.array(list_of_epsilons)[:,0]**2,linewidth=4,linestyle='--',label='-(lam-1)^2/(2*lam^2)*eps^2')
#         plt.xlabel('epsilon^2')
#         plt.ylabel('S1')
#         plt.title('S1 vs epsilon^2 lam='+str(lam))
#         plt.legend()
#         plt.tight_layout()
#         savefig('A_v_eps' + '.png', dpi=500)
#         plt.show()
#
#     if type(beta) is list:
#         A_aprox_theory, A_exact_theory, A_numerical = [], [], []
#         for path, lam in zip(guessed_paths, beta):
#             s0 = 1 / lam + np.log(lam) - 1
#             x0 = (lam - 1) / lam
#             y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, \
#             dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, lam, list_of_epsilons, tf, gamma)
#             q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
#
#             y1_for_linear = np.linspace(path[:, 0][-1], 0, 1000)
#             py1_linear = p1_star_clancy - ((p1_star_clancy - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
#             y2_for_linear = np.linspace(path[:, 1][-1], 0, 1000)
#             py2_linear = p2_star_clancy - ((p2_star_clancy - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
#             addition_to_path = np.stack((y1_for_linear, y2_for_linear, py1_linear, py2_linear), axis=1)
#
#             I_addition_to_path = simps(py1_linear, y1_for_linear) + simps(py2_linear, y2_for_linear)
#
#             path_addition = np.vstack((path, addition_to_path))
#
#             f_of_d = (1 / 2) * (lam) * (1 - list_of_epsilons ** 2)
#             D = (-1 + f_of_d + np.sqrt(list_of_epsilons ** 2 + f_of_d ** 2)) / (1 - list_of_epsilons ** 2)
#             I_pu_aprox = (((lam - 1) ** 3) / (4 * lam ** 3)) * list_of_epsilons ** 2
#             I_pw_aprox = -((lam + 1) * (lam - 1) ** 2 / (4 * lam ** 3)) * list_of_epsilons ** 2
#             A_aprox_theory.append(-I_pu_aprox + I_pw_aprox)
#             A_exact_theory_full = -(1 / 2) * (q_star[2] + q_star[3]) - (1 / lam) * D
#             A_exact_theory.append(A_exact_theory_full - s0)
#
#             # pudu=simps((path_new[:, 2] - path_new[:, 3]), ((path_new[:, 0] - path_new[:, 1] )/2))
#             # pwdw=simps((path_new[:, 2] + path_new[:, 3]), ((path_new[:, 0] + path_new[:, 1] )/2))
#             # pwdwcor=pwdw-s0
#             # A_numerical.append(pwdwcor+pudu)
#
#             # pudu_org=simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1] )/2))
#             # pwdw_org=simps((path[:, 2] + path[:, 3]), ((path[:, 0] + path[:, 1] )/2))
#             # pwdwcor_org=pwdw-s0
#             # A_numerical_org=pwdwcor_org+pudu_org
#
#             # pudu_add=simps((path_addition[:, 2] - path_addition[:, 3]), ((path_addition[:, 0] - path_addition[:, 1] )/2))
#             # pwdw_add=simps((path_addition[:, 2] + path_addition[:, 3]), ((path_addition[:, 0] + path_addition[:, 1] )/2))
#             # py1_add=simps(path_addition[:, 2], path_addition[:, 0])
#             # py2_add=simps((path_addition[:, 3], path_addition[:, 1]))
#
#             # pwdwcor_add=pwdw-s0
#             # A_numerical_add=pwdwcor_add+pudu_add
#
#             # generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:(p[:, 0] - p[:, 1]) / 2,lambda p,eps,x:p[:, 2] - p[:, 3]
#             #              ,'zw','u','pu',x0,lam,'p_u vs u','pu_vs_u',case_to_run,tf)
#
#             guessed_paths = [path_addition]
#             # generic_plot(guessed_paths,list_of_epsilons,lambda p,eps,alph:(p[:, 0] - p[:, 1]) / 2,lambda p,eps,x:p[:, 2] - p[:, 3]
#             #              ,'zw','u','pu',x0,lam,'p_u vs u','pu_vs_u_after',case_to_run,tf)
#
#             A_integration_y = simps(path[:, 2], path[:, 0]) + simps(path[:, 3], path[:, 1]) - s0 + I_addition_to_path
#             A_numerical.append(A_integration_y)
#         # fig = plt.figure()
#         # ax = fig.add_subplot(1, 1, 1)
#         plt.plot(np.array(beta), A_numerical, linewidth=4, linestyle='None', Marker='o',
#                  label='Numerical', markersize=10)
#         plt.plot(np.array(beta), A_exact_theory, linewidth=4, linestyle='None', Marker='^',
#                  label='Clancy', markersize=10)
#         plt.plot(np.array(beta), A_exact_theory, linewidth=4, linestyle='None', Marker='X',
#                  label='Miki', markersize=10)
#         beta_for_curve_function=np.linspace(1.01,beta[-1],1000)
#         f_of_lam=[-(l-1)**2/(2*l**2)*list_of_epsilons**2 for l in beta_for_curve_function]
#         plt.plot(beta_for_curve_function,f_of_lam, linewidth=4, linestyle='--',label='-(lam-1)^2/(2*lam^2)*eps^2')
#         plt.xlabel('lam')
#         plt.ylabel('S1')
#         plt.title('S1 vs lam epsilon=' + str(list_of_epsilons))
#         plt.legend()
#         plt.tight_layout()
#         savefig('A_v_lam' + '.png', dpi=500)
#         plt.show()
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        alpha=epsilon_mu/epsilon_lam
        pu_for_path=path[:,2]-path[:,3]
        u_for_path=(path[:,0]-path[:,1])/2
        u_thoery_alpha= alpha*(pu_for_path-2*x0*epsilon_lam)/(4*lam)
        u_function_pu=-((pu_for_path*(-2*epsilon_lam*(-1 + lam) +pu_for_path*lam)*(pu_for_path*lam-2*epsilon_lam*(1 + 2*lam)))/(4*(pu_for_path - 2*epsilon_lam)**3*lam**3))*epsilon_lam
        u_function_pu_linear=-epsilon_lam/(4*lam)
        u_theory_full= u_thoery_alpha+np.exp(-alpha)*u_function_pu
        integral_numeric=-(simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1] )/2)))
        integral_theory=(-(lam-1)**2/(2*lam**3))*(x0/2+alpha*(1-x0/2))*epsilon_lam**2
        plt.plot(pu_for_path/epsilon_lam,u_for_path/epsilon_lam,linewidth=4,label='Numerical eps='+str(epsilon))
        plt.plot(pu_for_path/epsilon_lam,u_theory_full/epsilon_lam,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('pu/eps')
    plt.ylabel('u/eps')
    plt.title('u/eps vs pu/eps, lam='+str(lam))
    plt.legend()
    plt.savefig('u_v_pu_thoery_both'+'.png',dpi=500)
    plt.show()

    A_numerical,A_theory,alpha_list,A_numerical_norm=[],[],[],[]
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        alpha=epsilon_mu/epsilon_lam
        alpha_list.append(alpha)
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy,\
        dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf,gamma)
        q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]

        y1_for_linear=np.linspace(path[:,0][-1],0,1000)
        py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
        y2_for_linear=np.linspace(path[:,1][-1],0,1000)
        py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear
        addition_to_path = np.stack((y1_for_linear,y2_for_linear,py1_linear,py2_linear),axis=1)

        I_addition_to_path=simps(py1_linear-py2_linear,(y1_for_linear-y2_for_linear)/2)
        pudu = simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1]) / 2))
        A_numerical.append(-(pudu+I_addition_to_path))
        A_numerical_norm.append(-(pudu+I_addition_to_path)/epsilon_lam**2)
        A_theory.append((-1)*alpha*(lam-1)**2/(2*lam**3)+np.exp(-alpha)*(-1+lam)**3/(4*lam**3))
    theory_line_for_plot=[-((eps[0]**2) * (-1 + lam)**2 *(1 + a - lam +a *lam))/(4 *lam**3) for eps,a in zip(list_of_epsilons,alpha_list)]
    alpha_for_thoery_plot = np.linspace(alpha_list[0], alpha_list[-1], 1000)
    # theory_line_for_plot_exp=np.array([(-1)*a*(lam-1)**2/(2*lam**3)+np.exp(-a)*(-1+lam)**3/(4*lam**3) for a in alpha_list])
    theory_line_for_plot_exp=np.array([(-1)*a*(lam-1)**2/(2*lam**3)+np.exp(-a)*(-1+lam)**3/(4*lam**3) for a in alpha_for_thoery_plot])
    plt.plot(alpha_list,A_numerical_norm,linewidth=4,linestyle='None', Marker='o', label='Numerical',markersize=10)
    # plt.plot(alpha_list,theory_line_for_plot,linewidth=4,linestyle='--', label='Theory',markersize=10)
    plt.plot(alpha_for_thoery_plot,theory_line_for_plot_exp,linewidth=4,linestyle='--', label='Theory',markersize=10)
    # plt.plot(alpha_list,A_theory,linewidth=4,linestyle='None',label='Theory', Marker='v',markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('Iu/eps^2')
    plt.title('Iu/eps^2 vs alpha lam='+str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig('pudu_v_eps' + '.png', dpi=500)
    plt.show()

    s0=-1+1/lam+np.log(lam)
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        alpha=epsilon_mu/epsilon_lam
        pw_for_path=path[:,2]+path[:,3]
        w_for_path=(path[:,0]+path[:,1])
        w_for_path_clancy=(path[:,0]+path[:,1])/2
        pw0_clancy = -2 * np.log(lam - 2 * w_for_path_clancy * (
                    1 + ((1 + alpha) / lam) * epsilon_mu ** 2) * lam) if case_to_run is 'la' else -2 * np.log(
            lam - 2 * w_for_path_clancy * (1 + alpha * ((1 + alpha) / lam) * epsilon_lam ** 2) * lam)

        pw_theory=-((epsilon_lam**2*((2*w_for_path*alpha*(1 + alpha)*lam)/(-1 + w_for_path) + (1 + (-1 + w_for_path)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)- 2*np.log(lam - w_for_path*lam)
        pw_theory_norm = -((((2*w_for_path*alpha*(1 + alpha)*lam)/(-1 + w_for_path) + (1 + (-1 + w_for_path)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)
        pw_theory_clancy = -((epsilon_lam ** 2 * (
                    (4 * w_for_path_clancy * alpha * (1 + alpha) * lam) / (-1 + 2 * w_for_path_clancy) + (
                        1 + (-1 + 2 * w_for_path_clancy) * lam) * (
                                1 + lam + 2 * alpha * lam))) / lam ** 2) - 2 * np.log(lam - w_for_path * lam)
        pw_theory_clancy_norm= -((((4*w_for_path_clancy*alpha*(1 + alpha)*lam)/(-1 + 2*w_for_path_clancy)+ (1 + (-1 + 2*w_for_path_clancy)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)
        y1_for_linear=np.linspace(path[:,0][-1],0,1000)
        py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
        y2_for_linear=np.linspace(path[:,1][-1],0,1000)
        py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear
        I_addition_to_path=simps(py1_linear+py2_linear,(y1_for_linear+y2_for_linear)/2)

        integral_numeric=simps(path[:, 2] + path[:, 3], (path[:, 0] + path[:, 1])/2)
        integral_numeric_correction=integral_numeric+I_addition_to_path-s0
        integral_theory=(-((lam-1)*(-1+lam*(lam+2*alpha*(-3-2*alpha+lam))))/(4*lam**3)-alpha*(1+alpha)*np.log(lam)/lam)*epsilon_lam**2
        # plt.plot(w_for_path,(pw_for_path-pw0)/epsilon_lam**2,linewidth=4,label='Numerical alpha='+str(alpha)+' eps='+str(epsilon))
        # plt.plot(w_for_path,(pw_theory-pw0)/epsilon_lam**2,linestyle='--',linewidth=4,label='Theory alpha='+str(alpha)+' eps='+str(epsilon))
        plt.plot(w_for_path_clancy, (pw_for_path - pw0_clancy) / epsilon_lam ** 2, linewidth=4,
                 label='Numerical eps=' + str(epsilon))        # plt.plot(w_for_path_clancy,((pw_for_path+(2*np.log(lam+2*lam*w_for_path_clancy)+(4*alpha*(w_for_path_clancy+w_for_path_clancy*alpha)/(lam+2*lam*w_for_path_clancy))*epsilon_lam**2))),linewidth=4,label='Numerical eps='+str(epsilon))
        plt.plot(w_for_path_clancy, (pw_theory_clancy - pw0_clancy) / epsilon_lam ** 2, linestyle='--', linewidth=4,
                 label='Theory eps=' + str(epsilon))
    plt.xlabel('w')
    plt.ylabel('(pw-pw0)/eps^2')
    plt.title('((pw-pw0)/eps^2 vs w, lam='+str(lam))
    plt.legend()
    plt.savefig('pw_vs_w_with_theory'+'.png',dpi=500)
    plt.show()

    A_numerical,A_theory,alpha_list,A_numerical_norm=[],[],[],[]
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        alpha=epsilon_mu/epsilon_lam
        alpha_list.append(alpha)

        y1_for_linear=np.linspace(path[:,0][-1],0,1000)
        py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
        y2_for_linear=np.linspace(path[:,1][-1],0,1000)
        py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear
        I_addition_to_path=simps(py1_linear+py2_linear,(y1_for_linear+y2_for_linear)/2)

        integral_numeric=simps(path[:, 2] + path[:, 3], (path[:, 0] + path[:, 1])/2)
        integral_numeric_correction=integral_numeric+I_addition_to_path-s0
        integral_theory=(-((lam-1)*(-1+lam*(lam+2*alpha*(-3-2*alpha+lam))))/(4*lam**3)-alpha*(1+alpha)*np.log(lam)/lam)*epsilon_lam**2
        A_numerical.append(integral_numeric_correction)
        A_numerical_norm.append(integral_numeric_correction/epsilon_lam**2)
        A_theory.append(integral_theory/epsilon_lam**2)
    alpha_for_thoery_plot = np.linspace(alpha_list[0], alpha_list[-1], 1000)
    theory_line_for_plot_norm=np.array([ (-((lam-1)*(-1+lam*(lam+2*a*(-3-2*a+lam))))/(4*lam**3)-a*(1+a)*np.log(lam)/lam) for a in alpha_for_thoery_plot])
    plt.plot(alpha_list,A_numerical_norm,linewidth=4,linestyle='None', Marker='o', label='Numerical',markersize=10)
    # plt.plot(alpha_list,A_theory,linewidth=4,linestyle='None',label='Theory marker', Marker='v',markersize=10)
    plt.plot(alpha_for_thoery_plot,theory_line_for_plot_norm,linewidth=4,linestyle='--',label='Theory')
    plt.xlabel('alpha')
    plt.ylabel('Iw/eps^2')
    plt.title('Iw/eps^2 vs alpha, lam='+str(lam))
    plt.legend()
    plt.savefig('Iw_v_eps_alpha'+'.png',dpi=500)
    plt.show()



    s0=-1+1/lam+np.log(lam)
    A_numerical,A_theory,alpha_list=[],[],[]
    for path, epsilon in zip(guessed_paths, list_of_epsilons):
        epsilon_lam, epsilon_mu = epsilon[0], epsilon[1]
        alpha=epsilon_mu/epsilon_lam
        alpha_list.append(alpha)

        y1_for_linear=np.linspace(path[:,0][-1],0,1000)
        py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
        y2_for_linear=np.linspace(path[:,1][-1],0,1000)
        py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear

        integral_numeric_w = simps(path[:, 2] + path[:, 3], (path[:, 0] + path[:, 1]) / 2)+simps(py1_linear+py2_linear,(y1_for_linear+y2_for_linear)/2)
        integral_numeric_u = simps((path[:, 2] - path[:, 3]), ((path[:, 0] - path[:, 1]) / 2))+simps(py1_linear-py2_linear,(y1_for_linear-y2_for_linear)/2)
        A_numerical.append((integral_numeric_w +integral_numeric_u-s0)/ epsilon_lam ** 2)

        I_addition_to_path_tot = simps(py1_linear, y1_for_linear) + simps(py2_linear, y2_for_linear)
        A_integration_y = simps(path[:, 2], path[:, 0]) + simps(path[:, 3], path[:, 1])-s0+I_addition_to_path_tot
        # A_numerical.append(A_integration_y/epsilon_lam**2)

        I_theory_u=-alpha*(lam-1)**2/(2*lam**3)+np.exp(-alpha)*(-1+lam)**3/(4*lam**3)
        I_theory_w = (-((lam-1)*(-1+lam*(lam+2*alpha*(-3-2*alpha+lam))))/(4*lam**3)-alpha*(1+alpha)*np.log(lam)/lam)

        A_theory.append(-I_theory_u+I_theory_w)


    alpha_for_thoery_plot = np.linspace(alpha_list[0], alpha_list[-1], 1000)
    theory_line_for_plot_norm=np.array([ -(-a*(lam-1)**2/(2*lam**3)+np.exp(-a)*(-1+lam)**3/(4*lam**3))+(-((lam-1)*(-1+lam*(lam+2*a*(-3-2*a+lam))))/(4*lam**3)-a*(1+a)*np.log(lam)/lam) for a in alpha_for_thoery_plot])

    plt.plot(alpha_list,A_numerical,linewidth=4,linestyle='None', Marker='o', label='Numerical',markersize=10)
    # plt.plot(alpha_list,A_theory,linewidth=4,linestyle='None',label='Theory marker', Marker='v',markersize=10)
    # plt.plot(alpha_list,A_theory,linestyle='None', Marker='v', label='Theory',markersize=10)
    plt.plot(alpha_for_thoery_plot,theory_line_for_plot_norm,linewidth=4,linestyle='--',label='Theory')
    plt.xlabel('alpha')
    plt.ylabel('S1/eps^2')
    plt.title('S1/eps^2 vs alpha, lam='+str(lam))
    plt.legend()
    plt.savefig('s1_vs_eps'+'.png',dpi=500)
    plt.show()



def plot_one_shot(angle_to_shoot,linear_combination,radius,time_vec,one_shot_dt,q_star,J,shot_dq_dt,beta):
    path = one_shot(angle_to_shoot, linear_combination,q_star,radius,time_vec,one_shot_dt,J,shot_dq_dt)
    plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='Numerical')
    plt.plot(path[:, 0] + path[:, 1],
             [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
             linewidth=4, linestyle='--', color='y', label='Theory')
    xlabel('y1+y2')
    ylabel('p1+p2')
    title('Theory vs numerical results')
    plt.legend()
    plt.scatter((path[:, 0][0] + path[:, 1][0], path[:, 0][-1] + path[:, 1][-1]),
                (path[:, 2][0] + path[:, 3][0], path[:, 2][-1] + path[:, 3][-1]), c=('g', 'r'), s=(100, 100))
    plt.show()
    return path

def man_find_best_div_path(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta):
    temp_best_div,r,shot_angle=best_diverge_path(shot_angle,radius,org_lin_combo,one_shot_dt,q_star, t0,J,shot_dq_dt)
    print(temp_best_div,' ', r, ' ',shot_angle)
    path = plot_one_shot(shot_angle,temp_best_div,r,t0 , one_shot_dt, q_star,J,shot_dq_dt,beta)
    return temp_best_div,r,shot_angle,path

def man_find_fine_tuning(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta):
    # temp_fine_tuning = fine_tuning(shot_angle,org_lin_combo,q_star,radius,t0,one_shot_dt,J,shot_dq_dt)
    temp_fine_tuning = fine_tuning(shot_angle,org_lin_combo,q_star,radius,t0,one_shot_dt,J,shot_dq_dt)
    print('linear combination= ',temp_fine_tuning,'radius=',radius, ' shot_angle=', shot_angle)
    path=plot_one_shot(shot_angle,temp_fine_tuning,radius,t0 ,one_shot_dt,q_star,J,shot_dq_dt,beta)
    return temp_fine_tuning


def plot_u_clancy_theory(path,epsilon,beta,gamma):
    epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
    w_for_path,u_for_path=(path[:,0]+path[:,1])/2,(path[:,0]-path[:,1])/2
    pu_theory = np.array([-np.log(1+(1-epsilon_lam)*z_w_u_space(w,u,epsilon_lam,beta,gamma))+np.log(1+(1+epsilon_lam)*z_w_u_space(w,u,epsilon_lam,beta,gamma)) for w,u in zip(w_for_path,u_for_path)])
    epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
    alpha, x0 = epsilon_mu / epsilon_lam, (lam - 1) / lam
    pu_theory_alpha = 2 * x0 * epsilon_lam + (4 * lam * u_for_path) / alpha
    u_integration_numerical = simps(path[:, 2] - path[:, 3], u_for_path)
    u_theory_integration = -(alpha * ((lam - 1) ** 2) * epsilon_lam ** 2) / (2 * math.pow(lam, 3))
    plt.plot(u_for_path, pu_theory_alpha, linewidth=4,
             linestyle=':', label='Theory Clancy=' + str(round(u_theory_integration, 5)))
    plt.plot(u_for_path,pu_theory,linestyle='--',linewidth=4)
    plt.legend()


def plot_numerical_only(shot_angle, lin_combo, one_shot_dt, radius, t0, q_star, J, shot_dq_dt, beta, case_to_run,epsilon,lam):
    epsilon_lam, epsilon_mu, s0 = epsilon[0], epsilon[1], 1 / lam - 1 + np.log(lam)
    lam = beta * (1 + epsilon_mu * epsilon_lam)
    path = one_shot(shot_angle, lin_combo, q_star, radius, t0, one_shot_dt, J, shot_dq_dt)
    I1, I2 = simps(path[:, 2], path[:, 0]), simps(path[:, 3], path[:, 1])
    A_integration = I1 + I2
    y1,y2,p1,p2 = path[:, 0],path[:, 1],path[:, 2],path[:, 3]
    delta_mu,delta_lam = 1 - epsilon_mu,1-epsilon_lam

    plt.plot(y1, p1, linewidth=4, label='Numerical, I= ' + str(round(I1, 4)))
    plt.xlabel('x1')
    plt.ylabel('p1')
    plt.scatter((q_star[0], 0), (0, q_star[2]), c=('g', 'r'), s=(100, 100))
    plt.title('p1 vs x1, lam=' + str(round(lam, 2)))
    plt.legend()
    plt.savefig('p1_v_x1_' + case_to_run + '.png', dpi=500)
    plt.show()

    plt.plot(y2, p2, linewidth=4, label='Numerical')
    plt.scatter((q_star[1], 0), (0, q_star[3]), c=('g', 'r'), s=(100, 100))
    plt.xlabel('x2')
    plt.ylabel('p2')
    plt.title('p2 vs x2, lam=' + str(round(lam, 2)))
    plt.legend()
    plt.savefig('p2_v_x2_' + case_to_run + '.png', dpi=500)
    plt.show()

    w, u, pw, pu = (y1 + y2) / 2, (y1 - y2) / 2, p1 + p2, p1 - p2
    plt.plot(w, pw, linewidth=4,
             label='Numerical Int=' + str(round((A_integration - s0) / delta_mu, 8)))
    plt.scatter(((q_star[1] + q_star[0]) / 2, 0), (0, q_star[2] + q_star[3]), c=('g', 'r'), s=(100, 100))
    plt.xlabel('w')
    plt.ylabel('pw')
    # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
    plt.title('pw vs w, lam=' + str(lam))
    plt.legend()
    plt.savefig('pw_v_w_' + case_to_run + '.png', dpi=500)
    plt.show()

    plt.plot(u, pu, linewidth=4,
             label='Numerical Int=' + str(round((I2 - I1) / delta_mu, 8)))
    plt.scatter(((q_star[0] - q_star[1]) / 2, 0), (0, q_star[2] - q_star[3]), c=('g', 'r'), s=(100, 100))
    plt.xlabel('u')
    plt.ylabel('pu')
    # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
    plt.title('pu vs u, lam=' + str(lam))
    plt.legend()
    plt.savefig('pu_v_u_' + case_to_run + '.png', dpi=500)
    plt.show()

    # m=y1-y2
    # pm=p1*(1+epsilon_lam)-p2*(1-epsilon_lam)
    # plt.plot(m, pm, linewidth=4,
    #          label='Numerical Int=' + str(round((I2 - I1) / delta_mu, 8)))
    # # plt.scatter(((q_star[0] - q_star[1]) / 2, 0), (0, q_star[2] - q_star[3]), c=('g', 'r'), s=(100, 100))
    # plt.xlabel('m')
    # plt.ylabel('pm')
    # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
    # plt.title('pm vs m, lam=' + str(lam))
    # plt.legend()
    # plt.savefig('pm_v_m_' + case_to_run + '.png', dpi=500)
    # plt.show()



    if case_to_run is 'dem':
        delta_mu=1-epsilon_mu
        action_theory = (1 - lam + lam*np.log(lam))/(2*lam) + (delta_mu*((-1 + lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)) - (2* (-1 + epsilon_lam)*(lam*(-1 + lam - lam*np.log(lam)) + (-3 + (5 - 2*lam)*lam + (-2 + lam)*lam*np.log(lam))*epsilon_lam))/(lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam))))/8

        plt.plot((y2 + y1) / 2, p1 + p2, linewidth=4,
                 label='Numerical Int=' + str(round((A_integration - s0 / 2) / delta_mu, 8)))
        plt.scatter(((q_star[1] + q_star[0]) / 2, 0), (0, q_star[2] + q_star[3]), c=('g', 'r'), s=(100, 100))
        plt.xlabel('w')
        plt.ylabel('pw')
        plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        plt.legend()
        plt.savefig('pw_v_w_' + case_to_run + '.png', dpi=500)
        plt.show()

        coef=(-1+epsilon_lam)/(2*(1+epsilon_lam))
        y2_for_theory=np.linspace(q_star[1],0,10000)
        # y2_theory= ((1 + (-1 + 2*y2_for_theory)*lam)*(-1 + epsilon_lam)*epsilon_lam)/(-lam + epsilon_lam*(-2 + (-2 + lam)*epsilon_lam))
        # p2_0_path=-np.log(lam*(1-2*y2*(1+coef*delta_mu)))
        p2_theory = ((-1 + epsilon_lam)*(y2_for_theory*lam + (-1 + 4*y2_for_theory)*(1 + (-1 + y2_for_theory)*lam)*epsilon_lam))/((-1 + 2*y2_for_theory)*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam))
        p2_0_path=-np.log(lam*(1-2*y2))
        p2_numerical_normalized=(p2-p2_0_path)/delta_mu

        # I2_theory_normalized= ((-1 +epsilon_lam)*(lam*(1 - lam +lam*np.log(lam)) +(3 - 5*lam +2*lam**2 -(-2 + lam)*lam*np.log(lam))*epsilon_lam))/(4*lam*(1 +epsilon_lam)*(-lam +(-2 + lam)*epsilon_lam))
        # I2_numerical_correction=(I2-s0/2)/delta_mu
        # I2_theory_normalized = ((-1 + lam)**2*(-1 + epsilon_lam)*epsilon_lam)/(4*lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam))
        I2_theory_normalized= ((-1 + epsilon_lam)*(lam*(1 - lam + lam*np.log(lam)) + (3 - 5*lam + 2*lam**2 - (-2 + lam)*lam*np.log(lam))*epsilon_lam))/(4*lam*(1 + epsilon_lam)*(-lam + (-2 + lam)*epsilon_lam))
        I2_numerical_correction = simps(p2_numerical_normalized, y2)

        plt.plot(y2, p2_numerical_normalized, linewidth=4,label='Numerical I= '+str(round(I2_numerical_correction,4)),linestyle='None',marker='.')
        plt.plot(y2_for_theory, p2_theory, linewidth=4,label='Theory, I= '+str(round(I2_theory_normalized,4)),linestyle='--')
        plt.scatter((q_star[1],0),(p2_numerical_normalized[0], (q_star[3]+np.log(lam))/delta_mu), c=('g', 'r'), s=(100, 100))
        plt.xlabel('y2')
        plt.ylabel('(p2-p2(0))/delta_mu')
        plt.title('(p2-p2(0))/delta_mu vs y2, lam=' + str(round(lam,2)))
        plt.legend()
        plt.savefig('p2_v_y2_norm_' + case_to_run + '.png', dpi=500)
        plt.show()

        y1_for_theory=np.linspace(q_star[0],0,10000)/delta_mu
        # y1_theory = (4*y1_for_theory*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)))/((-1 + lam)*delta_mu)

        p1_theory= ((1 + 4*(y1_for_theory) - lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)))/(-1 + lam)
        path_y1_norm=y1/delta_mu
        I1_norm=simps(p1, path_y1_norm)
        I1_norm_theory= ((-1 + lam)*np.log(2 - lam + (2*(-1 + lam))/(1 + epsilon_lam)))/ 8
        plt.plot(path_y1_norm, p1, linewidth=4,label='Numerical I ='+str(round(I1_norm,4)),linestyle='None',marker='.')
        plt.plot(y1_for_theory, p1_theory, linewidth=4,linestyle='--',label='Theory I= '+str(round(I1_norm_theory,4)))
        plt.xlabel('y1/delta_mu')
        plt.ylabel('p1')
        plt.scatter((q_star[0]/delta_mu,0),(0, q_star[2]), c=('g', 'r'), s=(100, 100))
        plt.title('p1 vs y1/delta, lam=' + str(round(lam,2)))
        plt.legend()
        plt.savefig('p1_v_y1_norm_' + case_to_run + '.png', dpi=500)
        plt.show()
    elif case_to_run is 'del':
        action_theory = (1 - lam + lam * np.log(lam)) /(2 *lam) +(delta_lam *(-1 +epsilon_mu)*((-1 + lam) ** 2 *lam +(3 - 4 * lam +lam ** 2 +2 * lam * np.log(lam))*epsilon_mu))/ (4 *lam *(1 +epsilon_mu)*(-lam +(-2 + lam) *epsilon_mu))

        plt.plot((y2 + y1) / 2, p1 + p2, linewidth=4,
                 label='Numerical Int=' + str(round((A_integration - s0 / 2) / delta_mu, 8)))
        plt.scatter(((q_star[1] + q_star[0]) / 2, 0), (0, q_star[2] + q_star[3]), c=('g', 'r'), s=(100, 100))
        plt.xlabel('w')
        plt.ylabel('pw')
        plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        plt.legend()
        plt.savefig('pw_v_w_' + case_to_run + '.png', dpi=500)
        plt.show()


        coef= -(((-1 +  epsilon_mu)* epsilon_mu)/((1 + epsilon_mu)*(-lam + (-2 + lam)*epsilon_mu)))
        y2_for_theory=np.linspace(q_star[1],0,10000)
        p2_theory = ((-1 + epsilon_mu)*(-1 + lam - 2*y2_for_theory*lam + (4*y2_for_theory*epsilon_mu)/((-1 + 2*y2_for_theory)*(-lam + (-2 + lam)*epsilon_mu))))/(2*(1 + epsilon_mu))
        p2_0_path=-np.log(lam*(1-2*y2))
        p2_numerical_normalized=(p2-p2_0_path)/delta_lam
        I2_theory_normalized= -((-1 + epsilon_mu)*(-((-1 + lam)**2*lam) + ((-1 + lam)*(6 + (-3 + lam)*lam)- 4*lam*np.log(lam))*epsilon_mu))/(8*lam*(1 + epsilon_mu)*(-lam + (-2 + lam)*epsilon_mu))
        I2_numerical_correction = simps(p2_numerical_normalized, y2)
        plt.plot(y2, p2_numerical_normalized, linewidth=4,label='Numerical I= '+str(round(I2_numerical_correction,4)),linestyle='None',marker='.')
        plt.plot(y2_for_theory, p2_theory, linewidth=4,label='Theory, I= '+str(round(I2_theory_normalized,4)),linestyle='--')
        plt.scatter((q_star[1],0),(p2_numerical_normalized[0], (q_star[3]+np.log(lam))/delta_lam), c=('g', 'r'), s=(100, 100))
        plt.xlabel('y2')
        plt.ylabel('(p2-p2(0))/delta_lam')
        plt.title('(p2-p2(0))/delta_lam vs y2, lam=' + str(round(lam,2)))
        plt.legend()
        plt.savefig('p2_v_y2_norm_' + case_to_run + '.png', dpi=500)
        plt.show()


        y1_for_theory=np.linspace(q_star[0],0,10000)
        p1_theory= ((1 - lam +2*y1_for_theory*(-2 + lam -2/(-1 +epsilon_mu))))/2
        path_p1_norm=p1/delta_lam
        I1_norm=simps(path_p1_norm, y1)

        I1_norm_theory = ((-1 + lam)**2*(-1 + epsilon_mu))/(-8*lam + 8*(-2 + lam)*epsilon_mu)

        plt.plot(y1, path_p1_norm, linewidth=4,label='Numerical I ='+str(round(I1_norm,4)),linestyle='None',marker='.')
        plt.plot(y1_for_theory, p1_theory, linewidth=4,linestyle='--',label='Theory I= '+str(round(I1_norm_theory,4)))
        plt.xlabel('y1')
        plt.ylabel('p1/delta_lam')
        plt.scatter((q_star[0],0),(0, q_star[2]/delta_lam), c=('g', 'r'), s=(100, 100))
        plt.title('p1 vs y1/delta, lam=' + str(round(lam,2)))
        plt.legend()
        plt.savefig('p1_v_y1_norm_' + case_to_run + '.png', dpi=500)
        plt.show()
    elif case_to_run is 'mu':
        w,u,pw,pu =(y1+y2)/2, (y1-y2)/2, p1+p2, p1-p2
        plt.plot(w, pw, linewidth=4,
                 label='Numerical Int=' + str(round((A_integration - s0) / delta_mu, 8)))
        plt.scatter(((q_star[1] + q_star[0]) / 2, 0), (0, q_star[2] + q_star[3]), c=('g', 'r'), s=(100, 100))
        plt.xlabel('w')
        plt.ylabel('pw')
        # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        plt.title('pw vs w, lam='+str(lam))
        plt.legend()
        plt.savefig('pw_v_w_' + case_to_run + '.png', dpi=500)
        plt.show()

        plt.plot(u, pu, linewidth=4,
                 label='Numerical Int=' + str(round((I2-I1) / delta_mu, 8)))
        plt.scatter(((q_star[0] - q_star[1]) / 2, 0), (0, q_star[2] - q_star[3]), c=('g', 'r'), s=(100, 100))
        plt.xlabel('u')
        plt.ylabel('pu')
        # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        plt.title('pu vs u, lam='+str(lam))
        plt.legend()
        plt.savefig('pu_v_u_' + case_to_run + '.png', dpi=500)
        plt.show()

        pw_0_path = -2*np.log(lam*(1-2*w))
        pw_norm=(pw-pw_0_path)/epsilon_mu
        
        pw_theory= (4*w*epsilon_lam)/(lam - 2*w*lam) + ((1 - (2*w*lam)/(-1 + lam))*epsilon_lam* (-2 + lam - lam*epsilon_lam**2 + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)))/np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4)
        
        plt.plot(w, pw_norm, linewidth=4,label='Numerical')
        plt.plot(w, pw_theory, linewidth=4,label='Theory',linestyle='--')
        plt.xlabel('w')
        plt.ylabel('(pw-pw(0)/eps_mu')
        # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        plt.title('(pw-pw(0))/eps_mu vs w, lam='+str(lam))
        plt.legend()
        plt.savefig('pw_v_w_norm' + case_to_run + '.png', dpi=500)
        plt.show()
    elif case_to_run is 'x':
        w,u,pw,pu =(y1+y2)/2, (y1-y2)/2, p1+p2, p1-p2
        # q1 =y1- np.array(
        #     [(1 - epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, 1.0) for x1, x2 in zip(path[:, 0], path[:, 1])])
        # q2 = y2 -np.array(
        #     [(1 + epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, 1.0) for x1, x2 in zip(path[:, 0], path[:, 1])])
        # pq1_0 = np.array([-np.log(1 + (1 - epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, 1.0)) for x1, x2 in
        #                   zip(path[:, 0], path[:, 1])])
        # pq2_0 = np.array([-np.log(1 + (1 + epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, 1.0)) for x1, x2 in
        #                   zip(path[:, 0], path[:, 1])])
        #
        # pq1_norm = (p1 - pq1_0)
        # plt.plot(q1, pq1_norm, linewidth=4, label='Numerical')
        # plt.xlabel('q1')
        # plt.ylabel('p1-pq1_clancy')
        # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        # plt.title('(p1-pq1_clancy) vs q1, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pq1_v_q_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        # pq2_norm = (p2 - pq2_0)
        # plt.plot(q2, pq2_norm, linewidth=4, label='Numerical')
        # plt.xlabel('q2')
        # plt.ylabel('p2-pq2_clancy')
        # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        # plt.title('(p2-pq2_clancy) vs q1, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pq2_v_q_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        # uq,wq=(q1-q2)/2,(q1+q2)/2
        # pqu_0,pqw_0= pq1_0-pq2_0,pq1_0+pq2_0
        # w,u,pw,pu =(y1+y2)/2, (y1-y2)/2, p1+p2, p1-p2
        #
        # pqu_norm = (pu- pqu_0)
        # plt.plot(uq, pqu_norm, linewidth=4, label='Numerical')
        # plt.xlabel('uq')
        # plt.ylabel('pu-puq')
        # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        # plt.title('(pu-puq) vs uq, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pqu_v_qu_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        # pqw_norm = (pw- pqw_0)
        # plt.plot(wq, pqw_norm, linewidth=4, label='Numerical')
        # plt.xlabel('wq')
        # plt.ylabel('pw-pwq')
        # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        # plt.title('(pw-pwq) vs wq, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pqw_v_qw_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        #
        # plt.plot(y1, pq1_norm, linewidth=4, label='Numerical')
        # plt.xlabel('y1')
        # plt.ylabel('p1-pq1_clancy')
        # plt.title('(p1-pq1_clancy) vs y1, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pq1_v_y1_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        # plt.plot(y2, pq2_norm, linewidth=4, label='Numerical')
        # plt.xlabel('y2')
        # plt.ylabel('p2-pq2_clancy')
        # plt.title('(p2-pq2_clancy) vs y2, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pq2_v_y2_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        # plt.plot(u, pqu_norm, linewidth=4, label='Numerical')
        # plt.xlabel('u')
        # plt.ylabel('pu-puq')
        # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        # plt.title('(pu-puq) vs u, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pqu_v_u_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
        #
        # plt.plot(w, pqw_norm, linewidth=4, label='Numerical')
        # plt.xlabel('w')
        # plt.ylabel('pw-pwq')
        # # plt.title('pw vs w, lam=' + str(round(lam, 2)) + ' s1=' + str(round((action_theory - s0 / 2) / delta_mu, 8)))
        # plt.title('(pw-pwq) vs w, lam=' + str(lam))
        # plt.legend()
        # plt.savefig('pqw_v_w_norm' + case_to_run + '.png', dpi=500)
        # plt.show()
            
        c1 = np.log(((lam + epsilon_lam*(2 -lam*epsilon_lam) + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4))*(1 + epsilon_lam*epsilon_mu))/(lam + 2*epsilon_lam - lam*epsilon_lam**2+ 2*epsilon_lam**2*epsilon_mu + np.sqrt(-4*(-1 + lam)*(-1 + epsilon_lam**2)*(1 + epsilon_lam*epsilon_mu)**2 + (2 - lam + lam*epsilon_lam**2 + 2*epsilon_lam*epsilon_mu)**2)))
        c2 = np.log(((lam - epsilon_lam*(2 + lam*epsilon_lam) + np.sqrt(lam**2 - 2*(-2 + lam**2)*epsilon_lam**2 + lam**2*epsilon_lam**4))*(1 + epsilon_lam*epsilon_mu))/(lam - 2*epsilon_lam - lam*epsilon_lam**2- 2*epsilon_lam**2*epsilon_mu + np.sqrt(-4*(-1 + lam)*(-1 + epsilon_lam**2)*(1 + epsilon_lam*epsilon_mu)**2 + (2 - lam + lam*epsilon_lam**2 + 2*epsilon_lam*epsilon_mu)**2)))

        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, J = eq_hamilton_J(sim, beta, epsilon, t,
                                                                                                 gamma)
        q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]

        # theory_path_theta1 = np.array(
        #     [-np.log(1 + (1 - epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, gamma)) for x1, x2 in
        #      zip(path[:, 0], path[:, 1])])
        # theory_path_theta2 = np.array(
        #     [-np.log(1 + (1 + epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, gamma)) for x1, x2 in
        #      zip(path[:, 0], path[:, 1])])

        # theory_path_theta1 = np.array(
        #     [-np.log(1 + (1 - epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, gamma)) for x1, x2 in
        #      zip(path[:, 0], path[:, 1])]) - c1
        # theory_path_theta2 = np.array(
        #     [-np.log(1 + (1 + epsilon_lam) * z_y1_y2(x1, x2, epsilon_lam, beta, gamma)) for x1, x2 in
        #      zip(path[:, 0], path[:, 1])]) - c2

        z_theory =np.array( [z_eps_mu(x1,x2,epsilon_lam,epsilon_mu,lam) for x1,x2 in zip(path[:,0],path[:,1])])

        y1_clancy_theory = np.array([y1_path_clancy(theta1,theta2,epsilon_mu,lam) for theta1,theta2 in zip(p1,p2)])
        y2_clancy_theory = np.array([y2_path_clancy(theta1,theta2,epsilon_mu,lam) for theta1,theta2 in zip(p1,p2)])


        theory_path_theta1 = np.array(
            [-np.log(1 + (1 - epsilon_lam) * ( z_eps_mu(x1,x2,epsilon_lam,epsilon_mu,lam) ) ) for x1, x2 in
             zip(path[:, 0], path[:, 1])])
        theory_path_theta2 = np.array(
            [-np.log(1 + (1 + epsilon_lam) * ( z_eps_mu(x1,x2,epsilon_lam,epsilon_mu,lam))) for x1, x2 in
             zip(path[:, 0], path[:, 1])])


        pw_theory = np.array([-np.log(
            1 + (1 - epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) - np.log(
            1 + (1 + epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) for w, u in
                              zip(w, u)])
        pu_theory = np.array([-np.log(
            1 + (1 - epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) + np.log(
            1 + (1 + epsilon_lam) * z_w_u_space(w, u, epsilon_lam, beta, gamma)) for w, u in
                              zip(w, u)])
        f_of_d = (1 / 2) * (beta / gamma) * (1 - epsilon_lam ** 2)
        D = (-1 + f_of_d + np.sqrt(epsilon_lam ** 2 + f_of_d ** 2)) / (1 - epsilon_lam ** 2)
        A_theory = -(1 / 2) * (q_star[2] + q_star[3]) - (gamma / beta) * D
        plt.plot(y1,p1,linewidth=4,label='Numerical')
        plt.plot(y1,theory_path_theta1,linewidth=4,label='Theory',linestyle='--')
        title('y1 vs p1 for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        xlabel('y1')
        ylabel('p1')
        plt.legend()
        plt.savefig('p1_v_y1_homo_theory' + '.png', dpi=500)
        plt.show()
        plt.plot(y2,p2,linewidth=4,label='Numerical')
        plt.plot(y2,theory_path_theta2,linewidth=4,label='Theory',linestyle='--')
        title('y2 vs p2 for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        xlabel('y2')
        ylabel('p2')
        plt.legend()
        plt.savefig('p2_v_y2_homo_theory' + '.png', dpi=500)
        plt.show()

        z1 =np.array( [(np.exp(-x) - 1) / (1 - epsilon_lam) for x in path[:, 2]])
        z2 =np.array( [(np.exp(-x) - 1) / (1 + epsilon_lam) for x in path[:, 3]])
        # z_theory =np.array( [z_y1_y2(x1,x2, epsilon_lam, lam, gamma) for x1,x2 in zip(path[:,0],path[:,1])])
        # z_theory =np.array( [z_eps_mu(x1,x2,epsilon_lam,epsilon_mu,lam) for x1,x2 in zip(path[:,0],path[:,1])])
        plt.plot(t, z1, linewidth=4, label='z for the 1-epsilon population')
        plt.plot(t, z2, linewidth=4, label='z for the 1+epsilon population', linestyle='--')
        plt.plot(t,z_theory,linewidth=4,linestyle=':',label='Theory homo')
        plt.scatter((t[0], t[-1]),
                    (z1[0], z2[-1]), c=('g', 'r'), s=(100, 100))
        xlabel('Time')
        ylabel('z')
        title('z vs Time for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        plt.legend()
        plt.savefig('z_v_time_both_case' + '.png', dpi=500)
        plt.show()

        plt.plot(p1,y1,linewidth=4, label='Numerical')
        plt.plot(p1,y1_clancy_theory,linewidth=4, label='Theory',linestyle='--')
        title('p1 vs y1 for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        xlabel('p1')
        ylabel('y1')
        plt.legend()
        plt.show()


        plt.plot(z_theory,w,linewidth=4, label='Numerical')
        plt.plot(z_theory,((-z_theory+lam+ (-1-z_theory)/(1+z_theory-z_theory*epsilon_lam**2))/(2*lam)),linewidth=4, label='Theory',linestyle='--')
        title('z vs w for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        xlabel('z')
        ylabel('w')
        plt.legend()
        plt.savefig('z_v_w'+'.png',dpi=500)
        plt.show()

        plt.plot(z_theory,u,linewidth=4, label='Numerical')
        title('z vs u for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        xlabel('z')
        ylabel('u')
        plt.legend()
        plt.savefig('z_v_u'+'.png',dpi=500)
        plt.show()


    elif case_to_run is 'bc':
        z_theory =np.array( [z_eps_mu(x1,x2,epsilon_lam,epsilon_mu,lam) for x1,x2 in zip(path[:,0],path[:,1])])
        y1_clancy_theory = np.array([y1_path_clancy(theta1,theta2,epsilon_mu,lam) for theta1,theta2 in zip(p1,p2)])
        y2_clancy_theory = np.array([y2_path_clancy(theta1,theta2,epsilon_mu,lam) for theta1,theta2 in zip(p1,p2)])
        plt.plot(p1,y1,linewidth=4, label='Numerical')
        plt.plot(p1,y1_clancy_theory,linewidth=4, label='Theory',linestyle='--')
        title('p1 vs y1 for lambda=' + str(round(lam,2)) + ' epsilon=' + str(epsilon))
        xlabel('p1')
        ylabel('y1')
        plt.legend()
        plt.show()


def man_div_path_and_fine_tuning(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta,case_to_run,epsilon,lam):
    lin_combo,r,shot_angle,path=man_find_best_div_path(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta)
    lin_combo=man_find_fine_tuning(shot_angle, r, t0, lin_combo, one_shot_dt, q_star, J, shot_dq_dt,beta)
    # plot_all_var(shot_angle, lin_combo, one_shot_dt, radius, t0, q_star, J, shot_dq_dt,beta,case_to_run,t0)
    plot_numerical_only(shot_angle, lin_combo, one_shot_dt, r, t0, q_star, J, shot_dq_dt, beta, case_to_run,epsilon,lam)


def plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,final_time_path,q_star,J,shot_dq_dt,beta,case_to_run,tf):
    lam=beta/gamma
    path = one_shot(shot_angle, lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    epsilon_theory = epsilon if type(epsilon) is float else epsilon[0]
    theory_path_theta1 = np.array([-np.log(1+(1-epsilon_theory)*z_y1_y2(x1,x2,epsilon_theory,beta,gamma)) for x1,x2 in zip(path[:,0],path[:,1])])
    theory_path_theta2 = np.array([-np.log(1+(1+epsilon_theory)*z_y1_y2(x1,x2,epsilon_theory,beta,gamma)) for x1,x2 in zip(path[:,0],path[:,1])])
    w_for_path,u_for_path=(path[:,0]+path[:,1])/2,(path[:,0]-path[:,1])/2
    pw_theory = np.array([-np.log(1+(1-epsilon_theory)*z_w_u_space(w,u,epsilon_theory,beta,gamma))-np.log(1+(1+epsilon_theory)*z_w_u_space(w,u,epsilon_theory,beta,gamma)) for w,u in zip(w_for_path,u_for_path)])
    pu_theory = np.array([-np.log(1+(1-epsilon_theory)*z_w_u_space(w,u,epsilon_theory,beta,gamma))+np.log(1+(1+epsilon_theory)*z_w_u_space(w,u,epsilon_theory,beta,gamma)) for w,u in zip(w_for_path,u_for_path)])
    f_of_d=(1/2)*(beta/gamma)*(1-epsilon_theory**2)
    D=(-1+f_of_d+np.sqrt(epsilon_theory**2+f_of_d**2))/(1-epsilon_theory**2)
    A_theory=-(1/2)*(q_star[2]+q_star[3])-(gamma/beta)*D
    A_integration = simps(path[:, 2],path[:, 0])+simps(path[:, 3],path[:, 1])
    A0=1/lam+np.log(lam)-1
    plt.plot(path[:, 0] + path[:, 1], path[:, 2] + path[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='Numerical for epsilon=' + str(epsilon))
    plt.plot(path[:, 0] + path[:, 1],
             [2 * np.log(gamma / (beta * (1 - (i + j)))) for i, j in zip(path[:, 0], path[:, 1])],
             linewidth=4, linestyle='--', color='y', label='Theory 1d homo')
    plt.plot(path[:, 0] + path[:, 1], theory_path_theta1+theory_path_theta2, linewidth=4,
             linestyle=':',  label='Theory Clancy=' + str(epsilon))
    xlabel('y1+y2')
    ylabel('p1+p2')
    title('pw vs w; eps='+str(epsilon)+' Lam='+str(round(beta))+' Action theory='+str(round(A_theory,4))+' Action int='+str(round(A_integration,4)))
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
                (q_star[2], q_star[3]), c=('m', 'k'), s=(100, 100))
    xlabel('Coordinate')
    ylabel('Momentum')
    title('y1,y2 vs p1,p2 for epsilon='+str(epsilon)+' and Lambda='+str(round(beta,1)))
    plt.legend()
    plt.savefig('p1p2_v_y1y2' + '.png', dpi=500)
    plt.show()
    plt.plot(path[:, 1]-path[:, 0], path[:, 3]-path[:, 2], linewidth=4,
             linestyle='None', Marker='.', label='y2-y1 vs p2-p1 for epsilon=' + str(epsilon))
    plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
    plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
                (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
    plt.scatter((q_star[1]-q_star[0] , 0 ),
                (0, q_star[3]-q_star[2]), c=('m', 'k'), s=(100, 100))
    xlabel('y2-y1')
    ylabel('p2-p1')
    title('p_u vs u for epsilon='+str(epsilon)+' and Lambda='+str(round(beta,1)))
    plt.legend()
    plt.savefig('pu_vs_y' + '.png', dpi=500)
    plt.show()

    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, \
    dq_dt_sus_inf, J = eq_hamilton_J(case_to_run, beta, epsilon, tf, gamma)
    q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
    y1_for_linear = np.linspace(path[:, 0][-1], 0, 1000)
    py1_linear = p1_star_clancy - ((p1_star_clancy - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
    y2_for_linear = np.linspace(path[:, 1][-1], 0, 1000)
    py2_linear = p2_star_clancy - ((p2_star_clancy - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear

    w_for_path,u_for_path=(path[:,0]+path[:,1])/2,(path[:,0]-path[:,1])/2


    addition_to_path = np.stack((y1_for_linear, y2_for_linear, py1_linear, py2_linear), axis=1)
    path_addition = np.vstack((path, addition_to_path))
    w_for_path_addition,u_for_path_addition=(path_addition[:,0]+path_addition[:,1])/2,(path_addition[:,0]-path_addition[:,1])/2


    plt.plot(w_for_path_addition, path_addition[:, 2]+path_addition[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
    pw_theory = np.array([-np.log(1+(1-epsilon_theory)*z_w_u_space(w,u,epsilon_theory,beta,gamma))-np.log(1+(1+epsilon_theory)*z_w_u_space(w,u,epsilon_theory,beta,gamma)) for w,u in zip(w_for_path_addition,u_for_path_addition)])
    plt.plot(w_for_path_addition,pw_theory,linestyle='--',linewidth=4)
    # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
    # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
    #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
    plt.scatter(((q_star[0]+q_star[1])/2 , 0 ),(0, (q_star[3]+q_star[2])), c=('g', 'r'), s=(100, 100))


    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        alpha, x0 = epsilon_lam / epsilon_mu, (lam - 1) / lam
        pw_theory_alpha = -2*np.log(lam*(1-2*w_for_path)) + (1-2*w_for_path*lam*(1-2*alpha/lam-1/lam**2))*epsilon_lam**2
        w_integration_numerical = simps(path[:, 2]+path[:, 3], w_for_path)
        s0=1/lam-1+np.log(lam)
        w_theory_integration=1/lam-1+np.log(lam)-((lam-1)*(1+2*alpha*lam+lam**2)/(4*math.pow(lam, 3)))*epsilon_lam**2
        numerical_correction=w_theory_integration-(1/lam-1+np.log(lam))
        theory_correction=w_integration_numerical-(1/lam-1+np.log(lam))
        pw0=[-2*np.log(lam*(1-2*w)) for w in w_for_path]
        plt.plot(w_for_path, pw_theory_alpha, linewidth=4,
                 linestyle=':', label='correction=' + str(round(theory_correction,5)))



    I_addition_to_path = simps(py1_linear + py2_linear, (y1_for_linear + y2_for_linear) / 2)


    w_integration_numerical = simps(path[:, 2] + path[:, 3], w_for_path)
    numerical_correction = w_integration_numerical - (1 / lam - 1 + np.log(lam))+I_addition_to_path
    xlabel('w')
    ylabel('pw')
    title('pw vs w eps='+str(epsilon)+' Lam='+str(beta)+ ' Int='+str(round(numerical_correction,5)))
    plt.legend()
    plt.savefig('pw_vs_w' + '.png', dpi=500)
    plt.show()


    I_addition_to_path = simps(py1_linear - py2_linear, (y1_for_linear - y2_for_linear) / 2)


    plt.plot((path_addition[:,0]-path_addition[:,1])/2, path_addition[:, 2]-path_addition[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
    # plt.plot(u_for_path,pu_theory,linestyle='--',linewidth=4)

    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        x0 = (lam - 1) / lam
        pu_theory_alpha = 2*x0*epsilon_lam+(4*lam*u_for_path)/alpha
        u_integration_numerical = simps(path[:, 2]-path[:, 3], u_for_path)
        u_theory_integration=-(alpha*((lam-1)**2)*epsilon_lam**2)/(2*math.pow(lam, 3))
        plt.plot(u_for_path, pu_theory_alpha, linewidth=4,
                 linestyle=':', label='correction=' + str(round(u_theory_integration,5)))



    # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
    # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
    #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
    u_integration_numerical = simps(path[:, 2] - path[:, 3], u_for_path)+I_addition_to_path
    plt.scatter(((q_star[0]-q_star[1])/2 , 0 ),
                (0, q_star[2]-q_star[3]), c=('g', 'r'), s=(100, 100))
    xlabel('u')
    ylabel('pu')
    title('p_u vs u for epsilon='+str(epsilon)+' and Lambda='+str(round(beta,1))+ ' Integration='+str(round(u_integration_numerical,5)))
    plt.legend()
    plt.savefig('pu_vs_u' + '.png', dpi=500)
    plt.show()

    w=w_for_path
    u=u_for_path


    # functions dealing with printing figure of clancy theory to second order
    z_analytical=(-2 + (-1 + 2*w)*(-1 + epsilon_theory**2)*lam + np.sqrt(4*epsilon_theory**2 - 8*u*epsilon_theory*(-1 + epsilon_theory**2)*lam + (1 - 2*w)**2*(-1 + epsilon_theory**2)**2*lam**2))/(2*(-1 + epsilon_theory**2))
    z_w_path=[z_w_u_space(w0,u0,epsilon_theory,beta,gamma) for w0,u0 in zip(w_for_path,u_for_path)]
    pu=path[:, 2]-path[:, 3]
    # pu_full_analytical = -np.log(-(-4 - 2*epsilon_theory+ lam -2*w*lam - epsilon_theory**2*lam +2*w*epsilon_theory**2*lam +np.sqrt(4*epsilon_theory**2 -8*u*epsilon_theory*(-1 + epsilon_theory**2)*lam+ (1 -2*w)**2*(-1 + epsilon_theory**2)** 2*lam**2))/(2.*(1 + epsilon_theory))) + np.log(1 +((1 + epsilon_theory)*(-2 +(-1 + 2*w)*(-1 + epsilon_theory**2)*lam+ np.sqrt(4*epsilon_theory**2 -8*u*epsilon_theory*(-1 + epsilon_theory**2)*lam+ (1 -2*w)**2*(-1 + epsilon_theory**2)**2*lam**2)))/(2*(-1 +epsilon_theory**2)))
    pu_theory_second_order=-epsilon_theory*(2-2*lam+4*lam*w_for_path)/(lam-2*w*lam)
    plt.plot(w_for_path, path[:, 2]-path[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='pu numerical, eps=' + str(epsilon))
    plt.plot(w_for_path,pu_theory,linestyle='--',linewidth=4,label='Exact clancy theory, eps='+str(epsilon_theory))
    plt.plot(w_for_path,pu_theory_second_order,linewidth=4,label='Approx clancy theory, eps='+str(epsilon_theory),linestyle=':')
    plt.xlabel('w')
    plt.ylabel('pu')
    plt.title('pu vs w for Lam='+str(lam))
    plt.legend()
    plt.savefig('second_order_pu_w'+'.png',dpi=500)
    plt.show()

    plt.plot(u/epsilon_lam,w)
    plt.xlabel('u/eps')
    plt.ylabel('w')
    plt.show()

    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        alpha, x0 = epsilon_lam / epsilon_mu, (lam - 1) / lam
        # pw_theory_alpha = -2*np.log(lam*(1-2*w_for_path)) + (1-2*w_for_path*lam*(1-2*alpha/lam-1/lam**2))*epsilon_lam**2
        w_integration_numerical = simps(path[:, 2]+path[:, 3], w_for_path)
        s0=1/lam-1+np.log(lam)
        w_theory_integration=1/lam-1+np.log(lam)-((lam-1)*(1+2*alpha*lam+lam**2)/(4*math.pow(lam, 3)))*epsilon_lam**2
        numerical_correction=w_theory_integration-(1/lam-1+np.log(lam))
        theory_correction=w_integration_numerical-(1/lam-1+np.log(lam))
        pw0=[-2*np.log(lam*(1-2*w)) for w in w_for_path]
        plt.plot(w_for_path, (path[:, 2]+path[:, 3]-pw0)/epsilon_lam**2, linewidth=4,
                 linestyle=':', label='correction=' + str(round(theory_correction,5)))


    xlabel('w')
    ylabel('pw')
    title('pw vs w eps='+str(epsilon)+' Lam='+str(beta)+ ' Int='+str(round(numerical_correction,5)))
    plt.legend()
    plt.savefig('pw_vs_w_end_program' + '.png', dpi=500)
    plt.show()


def plot_z(shot_angle, lin_combo,radius,final_time_path,one_shot_dt,beta,q_star,J,shot_dq_dt):
    path = one_shot(shot_angle, lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    epsilon_theory=epsilon if type(epsilon) is float else epsilon[0]
    z1=[(np.exp(-x)-1)/(1-epsilon_theory) for x in path[:,2]]
    z2=[(np.exp(-x)-1)/(1+epsilon_theory) for x in path[:,3]]
    # z_theory = [z(y1,y2) for y1,y2 in zip(path[:,0],path[:,1])]
    plt.plot(t,z1,linewidth=4,label='z for the 1-epsilon population')
    plt.plot(t,z2,linewidth=4,label='z for the 1+epsilon population',linestyle='--')
    # plt.plot(t,z_theory,linewidth=4,linestyle=':')
    plt.scatter((t[0], t[-1]),
                (z1[0], z2[-1]), c=('g', 'r'), s=(100, 100))
    xlabel('Time')
    ylabel('z')
    title('z vs Time for lambda='+str(beta)+' epsilon='+str(epsilon))
    plt.legend()
    plt.savefig('z_v_time' + '.png', dpi=500)
    plt.show()
    plt.plot(path[:,0],z1,linewidth=4,label='z for the 1-epsilon population')
    plt.plot(path[:,1],z2,linewidth=4,label='z for the 1+epsilon population',linestyle='--')
    xlabel('y')
    ylabel('z')
    title('The z=(exp(-p)-1)\(1-epsilon),(exp(-p)-1)\(1+epsilon)    ')
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
    title('z=(exp(-p)-1)\(1-epsilon),(exp(-p)-1)\(1+epsilon)')
    plt.legend()
    plt.savefig('z_v_p' + '.png', dpi=500)
    plt.show()

    w_for_path,u_for_path=(path[:,0]+path[:,1])/2,(path[:,0]-path[:,1])/2
    z_w_path=[z_w_u_space(w,u) for w,u in zip(w_for_path,u_for_path)]
    plt.plot(w_for_path,z_w_path,linewidth=4,label='epsilon='+str(epsilon))
    plt.scatter((w_for_path[0], w_for_path[-1]),
                (z_w_path[0], z_w_path[-1]), c=('g', 'r'), s=(100, 100))
    title('z(w,u) vs w for lambda='+str(beta)+' epsilon='+str(epsilon))
    xlabel('w')
    ylabel('z')
    plt.legend()
    plt.savefig('z_vs_w' + '.png', dpi=500)
    plt.show()

    plt.plot(u_for_path,z_w_path,linewidth=4,label='epsilon='+str(epsilon))
    plt.scatter((u_for_path[0], u_for_path[-1]),
                (z_w_path[0], z_w_path[-1]), c=('g', 'r'), s=(100, 100))
    title('z(w,u) vs u for lambda='+str(beta)+' epsilon='+str(epsilon))
    xlabel('u')
    ylabel('z')
    # title('z=(exp(-p)-1)\(1-epsilon),(exp(-p)-1)\(1+epsilon), lam=1.6 eps=0.5')
    plt.legend()
    plt.savefig('z_vs_u' + '.png', dpi=500)
    plt.show()

# def plot_eq_points(sim,beta,epsilon_matrix,t,gamma):
#
#     for case,epsilons in zip(sim,epsilon_matrix):
#         for eps in list_of_epsilons:
#             y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt,J = eq_hamilton_J(case,beta,eps,t,gamma)



if __name__=='__main__':
    #Network Parameters
    beta, gamma = 1.6, 1.0

    gamma=1.0
    # beta=[1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,4.5,5.0]

    abserr,relerr = 1.0e-20,1.0e-13
    # list_of_epsilons=[(0.9,0.02),(0.9,0.04),(0.9,0.06),(0.9,0.08),(0.9,0.1)]
    list_of_epsilons=[(0.5,0.1),(0.6,0.1),(0.7,0.1),(0.8,0.1),(0.9,0.1)]
    # list_of_epsilons=[(0.002,0.1)]
    # list_of_epsilons = [(0.5, 0.05)]
    # list_of_epsilons=0.1
    sim='x'

    # A way to confirm the hamiltion's numericaly
    # Jacobian_H = ndft.Jacobian(H)
    # dq_dt_numerical = lambda q: np.multiply(Jacobian_H(q),np.array([-1,-1,1,1]).reshape(1,4))

    # ODE parameters
    stoptime=20.0
    numpoints = 10000


    # Create the time samples for the output of the ODE solver
    t = np.linspace(0.0,stoptime,numpoints)

    dt=16.0/(numpoints-1)

    # Radius around eq point,Time of to advance the self vector
    # r002=2e-07
    # r001=4e-7
    # r=2.56e-06
    r=1.6384e-06
    angle=0.04239816339744822

    epsilon=(0.02,0.1)
    #lin002=0.9999930516412242
    #int_lin_combo001=0.9999658209936237
    # int_lin_combolam5=0.9999658419290037
    # int_lin_comboeps(01,01)=1.0001955976196242
    # int_lin_combo0018e-7=0.9999657791228237
    # int_lin_combo_all_runs=1.0002181438489302
    # int_lin_combo=1.0259660334473293
    # int_lin_combo=0.7386390749669806
    int_lin_combo=1.000040262472682
    # int_lin_combo=1.000040262472682

    # int_lin_combo=1.001321728340301
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf,J=eq_hamilton_J(sim, beta, epsilon, t, gamma)
    q_star=[y1_0, y2_0,  p1_star_clancy, p2_star_clancy]
    # man_div_path_and_fine_tuning(-np.pi/2,r,t,0.9920007999,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim)
    # man_div_path_and_fine_tuning(-np.pi/2,r,t,0.9920007999,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim)
    # man_div_path_and_fine_tuning(0.04239816339744822,r,t,1.000040262472682,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim,epsilon,beta/gamma)

    # man_div_path_and_fine_tuning(0.04239816339744822,r,t,1.000280636141402,dt,q_star,J,dq_dt_sus_inf,beta/(1-epsilon[0]*epsilon[1]),sim,epsilon,beta/gamma)


    # man_div_path_and_fine_tuning(0.04239816339744822,r,t,1.0006148654602816,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim,epsilon,beta/gamma)

    # man_div_path_and_fine_tuning(0.04239816339744822,r,t,1.0001014251271045,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim,epsilon,beta/gamma)


    # plot_one_shot(np.pi/4-0.62, 1.0081923932, r, t, dt, q_star, J, dq_dt_sus_inf, beta)

    # man_div_path_and_fine_tuning(np.pi/4-0.785084,r,t,1.0001758373880196,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim)
    # man_div_path_and_fine_tuning(np.pi/4-0.785084,r,t,0.9999758373880197,dt,q_star,J,dq_dt_sus_inf,beta/(1+epsilon[0]*epsilon[1]),sim)
    # man_div_path_and_fine_tuning(np.pi / 4 - 0.74, r, t, 0.7386390749669806, dt, q_star, J, dq_dt_sus_inf,
    #                              beta / (1 + epsilon[0] * epsilon[1]), sim)
    # multi_eps_normalized_path(sim,)
    # eq_points_alpha(epsilon, beta, gamma)
    # multi_eps_normalized_path(sim, list_of_epsilons, beta, gamma, numpoints, dt, r,int_lin_combo)
    # temp=multi_eps_normalized_path(sim, list_of_epsilons, beta, gamma, numpoints, dt, r, int_lin_combo)
    # plot_eps_mu_clancy(temp, beta, gamma, list_of_epsilons, sim, t)
    # plot_eps_mu_sub(temp,beta,gamma,list_of_epsilons,sim,t)
    # plot_eps_lam_sub(temp,beta,gamma,list_of_epsilons,sim,t)

    # sim=['al','la']
    # sim=['x','x']
    # epsilon_matrix=[[(0.8,0.03),(0.8,0.06),(0.8,0.1),(0.8,0.13),(0.8,0.16)],[(0.03,0.8),(0.06,0.8),(0.1,0.8),(0.13,0.8),(0.16,0.8)]]
    # # epsilon_matrix = [[(0.1, 0.02)],
    # #                   [(0.02, 0.1)]]
    # sim_paths=[]
    # for case,epsilons in zip(sim,epsilon_matrix):
    #     sim_paths.append(multi_eps_normalized_path(case, epsilons, beta, gamma, numpoints, dt, r, int_lin_combo))
    # plot_multi_sim_path(sim_paths, beta, gamma, epsilon_matrix, sim, t)
    # eq_points_exact(epsilon,beta,gamma)

    #
    sim=['x','x']
    # epsilon_matrix=[[(0.02,0.05),(0.04,0.05),(0.06,0.05),(0.08,0.05),(0.1,0.05),(0.14,0.05),(0.18,0.05),(0.22,0.05),(0.26,0.05),(0.3,0.05),(0.36,0.05),(0.4,0.05),(0.45,0.05),(0.5,0.05),(0.55,0.05),(0.6,0.05),(0.65,0.05),(0.7,0.05),(0.75,0.05),(0.8,0.05),(0.85,0.05),(0.9,0.05),(0.93,0.05),(0.94,0.05),(0.98,0.05)]]
    epsilon_matrix=[[(0.5,0.1),(0.1,0.1)],[(0.5,0.05),(0.1,0.05)]]
    # epsilon_matrix=[[(e,0.1) for e in np.linspace(0.5,0.6,2)]]
    # # epsilon_matrix = [[(0.1, 0.02)],
    # #                   [(0.02, 0.1)]]
    sim_paths=[]
    for case,epsilons in zip(sim,epsilon_matrix):
        sim_paths.append(multi_eps_normalized_path(case, epsilons, beta, gamma, numpoints, dt, r, int_lin_combo,angle))
    # plot_deltas(sim_paths,epsilon_matrix,[lambda p,eps,l:p[:,1],lambda p,eps,l:p[:,0]/(1-eps[1])] ,[lambda p,eps,l:(p[:,3]+np.log(l*(1-2*p[:,1])))/(1-eps[1]),lambda p,eps,l:p[:,2]],'dem',['p2','p1'],
    #              ['y2','y1/delta_mu'],['(p2-p2(0))/delta_mu','p1'],beta/gamma,['(p2-p2(0))/delta_mu vs y2','p1 vs y1/delta'],['p2_norm_v_y2','p1_norm_v_y1'],labeladdon=lambda x,y:'')
    # plot_deltas(sim_paths,epsilon_matrix,[lambda p,eps,l:p[:,1],lambda p,eps,l:p[:,0]] ,[lambda p,eps,l:(p[:,3]+np.log(l*(1-2*p[:,1])))/(1-eps[0]),lambda p,eps,l:p[:,2]/(1-eps[0])],'del',['pl2','pl1'],
    #              ['y2','y1'],['(p2-p2(0))/delta_mu','p1/delta_lam'],beta/gamma,['(p2-p2(0))/delta_mu vs y2','p1/delta_lam vs y1'],['pl2_norm_v_y2','pl1_norm_v_y1'],labeladdon=lambda x,y:'')
    # plot_integation(sim_paths, epsilon_matrix,beta/gamma,'del')
    plot_integration_theory_z(sim_paths, epsilon_matrix,sim ,beta, gamma,'s')


    # plot_eq_points(sim,beta,epsilon_matrix,t,gamma)
