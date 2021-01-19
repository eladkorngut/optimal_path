import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
from pylab import figure, plot, xlabel, grid, legend, title,savefig,ylabel
import numdifftools as ndft


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


def eq_hamilton_J(case_to_run,beta,epsilon=0.0,t=None,gamma=1.0):
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
        epsilon_lam,epsilon_mu=epsilon[0],epsilon[1]
        dy1_dt_sus_inf = lambda q: beta * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 - epsilon_mu) * (
                    1 / 2 - q[0]) * np.exp(q[2]) - gamma * q[0] * np.exp(-q[2])
        dy2_dt_sus_inf = lambda q: beta * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 + epsilon_mu) * (
                    1 / 2 - q[1]) * np.exp(q[3]) - gamma * q[1] * np.exp(-q[3])
        dtheta1_dt_sus_inf = lambda q: -beta * (1 - epsilon_lam) * (
                    (1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 + epsilon_mu) * (1 / 2 - q[1]) * (
                        np.exp(q[3]) - 1)) + beta * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (
                                                   1 - epsilon_mu) * (np.exp(q[2]) - 1) - gamma * (np.exp(-q[2]) - 1)
        dtheta2_dt_sus_inf = lambda q: -beta * (1 + epsilon_lam) * (
                    (1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 + epsilon_mu) * (1 / 2 - q[1]) * (
                        np.exp(q[3]) - 1)) + beta * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (
                                                   1 + epsilon_mu) * (np.exp(q[3]) - 1) - gamma * (np.exp(-q[3]) - 1)
        dq_dt_sus_inf = lambda q, t=None: np.array(
            [dy1_dt_sus_inf(q), dy2_dt_sus_inf(q), dtheta1_dt_sus_inf(q), dtheta2_dt_sus_inf(q)])
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_inf_only(epsilon, beta, gamma)
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

def best_diverge_path(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    org_div_time = when_path_diverge(path)
    going_up=path[:,0][int(org_div_time)-2]+path[:,1][int(org_div_time)-2]>=path[:,0][0]+path[:,0][0]
    while going_up:
        radius=radius*2
        path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        org_div_time = when_path_diverge(path)
        going_up = path[:, 0][int(org_div_time) - 2] + path[:, 1][int(org_div_time) - 2] >= path[:, 0][0] + path[:, 0][0]
    dl = 0.1
    path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    lin_combo=org_lin_combo
    while path_diverge(path) is True:
        org_div_time = when_path_diverge(path)
        lin_combo_step_up=lin_combo+dl
        lin_combo_step_down=lin_combo-dl
        path_up=one_shot(shot_angle,lin_combo_step_up,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        path_down=one_shot(shot_angle,lin_combo_step_down,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
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
            lin_combo=lin_combo+dl
        path = one_shot(shot_angle,lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    return lin_combo,radius

def fine_tuning(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt):
    min_accurecy,dl=0.001,1e-4
    path = one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    distance_from_theory = lambda p:np.sqrt(((p[:,0][-1]-p[:,1][-1])/2)**2+(q_star[2]-q_star[3]-(p[:,2][-1]-p[:,3][-1]))**2)
    lin_combo = org_lin_combo
    current_distance = distance_from_theory(path)
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
        lin_combo, radius = best_diverge_path(shot_angle, radius,lin_combo,one_shot_dt,q_star,np.linspace(0.0,s,sample_size),J,shot_dq_dt )
    lin_combo, radius = best_diverge_path(shot_angle, org_radius, lin_combo, one_shot_dt, q_star,np.linspace(0.0, sampleingtime[-1], sample_size), J, shot_dq_dt)
    lin_combo= fine_tuning(shot_angle, lin_combo,q_star,radius, np.linspace(0.0,sampleingtime[-1],sample_size), one_shot_dt,J,shot_dq_dt)
    # plot_one_shot(shot_angle,lin_combo,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt)
    # plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,np.linspace(0.0,sampleingtime[-1],sample_size))
    return lin_combo,radius, one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt,J,shot_dq_dt)

def guess_path_lam(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,org_radius,sample_size,J,shot_dq_dt,beta):
    radius=org_radius
    for s in sampleingtime:
        lin_combo, radius = best_diverge_path(shot_angle, radius,lin_combo,one_shot_dt,q_star,np.linspace(0.0,s,sample_size),J,shot_dq_dt)
        lin_combo = fine_tuning(shot_angle, lin_combo, q_star, radius, np.linspace(0.0, s, sample_size),
                                one_shot_dt, J, shot_dq_dt)
        path=one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,s,sample_size),one_shot_dt,J,shot_dq_dt)
        if path[:,0][-1]+path[:,1][-1]<1e-3:break
    # lin_combo, radius = best_diverge_path(shot_angle, org_radius, lin_combo, one_shot_dt, q_star,np.linspace(0.0, sampleingtime[-1], sample_size), J, shot_dq_dt)
    # lin_combo= fine_tuning(shot_angle, lin_combo,q_star,radius, np.linspace(0.0,sampleingtime[-1],sample_size), one_shot_dt,J,shot_dq_dt)
    # plot_one_shot(shot_angle,lin_combo,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt)
    # plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,np.linspace(0.0,sampleingtime[-1],sample_size))
    return lin_combo,radius, one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,s,sample_size),one_shot_dt,J,shot_dq_dt)


def multi_eps_normalized_path(case_to_run,list_of_epsilons,beta,gamma,numpoints,one_shot_dt,radius,lin_combo=1.00008204478397):
    guessed_paths=[]
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
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt,J = eq_hamilton_J(case_to_run,beta,eps,t,gamma)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            lin_combo,temp_radius,path=guess_path(sampleingtime,np.pi/4-0.785084,lin_combo,q_star,one_shot_dt,radius,numpoints,J,shot_dq_dt)
            guessed_paths.append(path)
    plot_multi_guessed_paths(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,np.linspace(0,sampleingtime[-1],numpoints))
    return guessed_paths

def plot_multi_guessed_paths(guessed_paths,beta,gamma,list_of_epsilons,case_to_run,tf):
    if type(beta) is not list:
        lam=beta/gamma
        x0=(lam - 1) / lam
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

    theory_line_for_plot=[-((eps[0]**2) * (-1 + lam)**2 *(1 + a - lam +a *lam))/(4 *lam**3) for eps,a in zip(list_of_epsilons,alpha_list)]
    theory_line_for_plot_exp=np.array([(-1)*a*(lam-1)**2/(2*lam**3)+np.exp(-a)*(-1+lam)**3/(4*lam**3) for a in alpha_list])
    plt.plot(alpha_list,A_numerical_norm,linewidth=4,linestyle='None', Marker='o', label='Numerical',markersize=10)
    # plt.plot(alpha_list,theory_line_for_plot,linewidth=4,linestyle='--', label='Theory',markersize=10)
    plt.plot(alpha_list,theory_line_for_plot_exp,linewidth=4,linestyle='--', label='Theory',markersize=10)
    # plt.plot(alpha_list,A_theory,linewidth=4,linestyle='None',label='Theory', Marker='v',markersize=10)
    plt.xlabel('alpha')
    plt.ylabel('Iu/eps^2')
    plt.title('I_u vs alpha lam='+str(lam))
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
        pw0=- 2*np.log(lam - w_for_path*(1+alpha*((1+alpha)/lam)*epsilon_lam**2)*lam)
        pw0_clancy=-2*np.log(lam - 2*w_for_path_clancy*(1+alpha*((1+alpha)/lam)*epsilon_lam**2)*lam)
        pw_theory=-((epsilon_lam**2*((2*w_for_path*alpha*(1 + alpha)*lam)/(-1 + w_for_path) + (1 + (-1 + w_for_path)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)- 2*np.log(lam - w_for_path*lam)
        pw_theory_norm = -((((2*w_for_path*alpha*(1 + alpha)*lam)/(-1 + w_for_path) + (1 + (-1 + w_for_path)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)
        pw_theory_clancy= -((epsilon_lam**2*((4*w_for_path_clancy*alpha*(1 + alpha)*lam)/(-1 + 2*w_for_path_clancy)+ (1 + (-1 + 2*w_for_path_clancy)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)- 2*np.log(lam - w_for_path*lam)
        pw_theory_clancy_norm= -((((4*w_for_path_clancy*alpha*(1 + alpha)*lam)/(-1 + 2*w_for_path_clancy)+ (1 + (-1 + 2*w_for_path_clancy)*lam)*(1 + lam + 2*alpha*lam)))/lam**2)


        y1_for_linear=np.linspace(path[:,0][-1],0,1000)
        py1_linear=p1_star_clancy-((p1_star_clancy-path[:,2][-1])/path[:,0][-1])*y1_for_linear
        y2_for_linear=np.linspace(path[:,1][-1],0,1000)
        py2_linear=p2_star_clancy-((p2_star_clancy-path[:,3][-1])/path[:,1][-1])*y2_for_linear
        I_addition_to_path=simps(py1_linear+py2_linear,(y1_for_linear+y2_for_linear))

        integral_numeric=(1/2)*simps(path[:, 2] + path[:, 3], (path[:, 0] + path[:, 1]))
        integral_numeric_correction=integral_numeric+I_addition_to_path-s0
        integral_theory=(-((lam-1)*(-1+lam*(lam+2*alpha*(-3-2*alpha+lam))))/(4*lam**3)-alpha*(1+alpha)*np.log(lam)/lam)*epsilon_lam**2
        # plt.plot(w_for_path,(pw_for_path-pw0)/epsilon_lam**2,linewidth=4,label='Numerical alpha='+str(alpha)+' eps='+str(epsilon))
        # plt.plot(w_for_path,(pw_theory-pw0)/epsilon_lam**2,linestyle='--',linewidth=4,label='Theory alpha='+str(alpha)+' eps='+str(epsilon))
        plt.plot(w_for_path_clancy,(pw_for_path-pw0_clancy)/epsilon_lam**2+2*alpha,linewidth=4,label='Numerical eps='+str(epsilon))
        # plt.plot(w_for_path_clancy,((pw_for_path+(2*np.log(lam+2*lam*w_for_path_clancy)+(4*alpha*(w_for_path_clancy+w_for_path_clancy*alpha)/(lam+2*lam*w_for_path_clancy))*epsilon_lam**2))),linewidth=4,label='Numerical eps='+str(epsilon))
        plt.plot(w_for_path_clancy,(pw_theory_clancy-pw0_clancy)/epsilon_lam**2,linestyle='--',linewidth=4,label='Theory eps='+str(epsilon))
    plt.xlabel('w')
    plt.ylabel('(pw-pw0)/eps^2')
    plt.title('((pw-pw0)/eps^2 vs pw, lam='+str(lam))
    plt.legend()
    plt.savefig('pw_vs_w_with_theory'+'.png',dpi=500)
    plt.show()
    print('Integral Numeric = '+str(integral_numeric_correction)+' | Theory = '+str(integral_theory))


def plot_one_shot(angle_to_shoot,linear_combination,radius,time_vec,one_shot_dt,q_star,J,shot_dq_dt,beta):
    path = one_shot(angle_to_shoot, linear_combination,q_star,radius,time_vec,one_shot_dt,J,shot_dq_dt)
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

def man_find_best_div_path(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta):
    temp_best_div,r=best_diverge_path(shot_angle,radius,org_lin_combo,one_shot_dt,q_star, t0,J,shot_dq_dt)
    print(temp_best_div,' ', r)
    path = plot_one_shot(shot_angle,temp_best_div,r,t0 , one_shot_dt, q_star,J,shot_dq_dt,beta)
    return temp_best_div,r,path

def man_find_fine_tuning(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta):
    temp_fine_tuning = fine_tuning(shot_angle,org_lin_combo,q_star,radius,t0,one_shot_dt,J,shot_dq_dt)
    print(temp_fine_tuning)
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


def man_div_path_and_fine_tuning(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta):
    lin_combo,r,path=man_find_best_div_path(shot_angle,radius,t0,org_lin_combo,one_shot_dt,q_star,J,shot_dq_dt,beta)
    lin_combo=man_find_fine_tuning(shot_angle, r, t0, lin_combo, one_shot_dt, q_star, J, shot_dq_dt,beta)
    # plot_all_var(shot_angle, lin_combo, one_shot_dt, radius, t0, q_star, J, shot_dq_dt,beta)


def plot_all_var(shot_angle,lin_combo,one_shot_dt,radius,final_time_path,q_star,J,shot_dq_dt,beta):
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
    title('pw vs w; eps='+str(epsilon)+' Lam='+str(beta)+' Action theory='+str(round(A_theory,4))+' Action int='+str(round(A_integration,4)))
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
    title('y1,y2 vs p1,p2 for epsilon='+str(epsilon)+' and Lambda='+str(beta))
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
    title('p_u vs u for epsilon='+str(epsilon)+' and Lambda='+str(beta))
    plt.legend()
    plt.savefig('pu_vs_y' + '.png', dpi=500)
    plt.show()
    plt.plot(w_for_path, path[:, 2]+path[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
    plt.plot(w_for_path,pw_theory,linestyle='--',linewidth=4)
    # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
    # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
    #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
    plt.scatter(((q_star[0]+q_star[1])/2 , 0 ),(0, (q_star[3]+q_star[2])), c=('g', 'r'), s=(100, 100))


    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        alpha, x0 = epsilon_mu / epsilon_lam, (lam - 1) / lam
        pw_theory_alpha = -2*np.log(lam*(1-2*w_for_path)) + (1-2*w_for_path*lam*(1-2*alpha/lam-1/lam**2))*epsilon_lam**2
        w_integration_numerical = simps(path[:, 2]+path[:, 3], w_for_path)
        s0=1/lam-1+np.log(lam)
        w_theory_integration=1/lam-1+np.log(lam)-((lam-1)*(1+2*alpha*lam+lam**2)/(4*math.pow(lam, 3)))*epsilon_lam**2
        numerical_correction=w_theory_integration-(1/lam-1+np.log(lam))
        theory_correction=w_integration_numerical-(1/lam-1+np.log(lam))
        pw0=[-2*np.log(lam*(1-2*w)) for w in w_for_path]
        plt.plot(w_for_path, pw_theory_alpha, linewidth=4,
                 linestyle=':', label='correction=' + str(round(theory_correction,5)))


    xlabel('w')
    ylabel('pw')
    title('pw vs w eps='+str(epsilon)+' Lam='+str(beta)+ ' Int='+str(round(numerical_correction,5)))
    plt.legend()
    plt.savefig('pw_vs_w' + '.png', dpi=500)
    plt.show()
    plt.plot(u_for_path, path[:, 2]-path[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
    plt.plot(u_for_path,pu_theory,linestyle='--',linewidth=4)

    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        alpha, x0 = epsilon_mu / epsilon_lam, (lam - 1) / lam
        pu_theory_alpha = 2*x0*epsilon_lam+(4*lam*u_for_path)/alpha
        u_integration_numerical = simps(path[:, 2]-path[:, 3], u_for_path)
        u_theory_integration=-(alpha*((lam-1)**2)*epsilon_lam**2)/(2*math.pow(lam, 3))
        plt.plot(u_for_path, pu_theory_alpha, linewidth=4,
                 linestyle=':', label='correction=' + str(round(u_theory_integration,5)))



    # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
    # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
    #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
    plt.scatter(((q_star[0]-q_star[1])/2 , 0 ),
                (0, q_star[2]-q_star[3]), c=('g', 'r'), s=(100, 100))
    xlabel('u')
    ylabel('pu')
    title('p_u vs u for epsilon='+str(epsilon)+' and Lambda='+str(beta)+ ' Integration='+str(round(u_integration_numerical,5)))
    plt.legend()
    plt.savefig('pu_vs_u' + '.png', dpi=500)
    plt.show()

    w=w_for_path
    u=u_for_path



    plt.plot(w_for_path, path[:, 2]+path[:, 3], linewidth=4,
             linestyle='None', Marker='.', label='w vs pw for epsilon=' + str(epsilon))
    plt.plot(w_for_path,pw_theory,linestyle='--',linewidth=4)
    # plt.plot(path[:, 1]-path[:, 0], theory_path_theta2-theory_path_theta1, linewidth=4,linestyle='--',label='Theory Clancy')
    # plt.scatter((path[:, 1][0]-path[:, 0][0] , path[:, 1][-1]-path[:, 0][-1] ),
    #             (path[:, 3][0]-path[:, 2][0], path[:, 3][-1]-path[:, 2][-1]), c=('g', 'r'), s=(100, 100))
    plt.scatter(((q_star[0]+q_star[1])/2 , 0 ),(0, (q_star[3]+q_star[2])), c=('g', 'r'), s=(100, 100))


    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        alpha, x0 = epsilon_mu / epsilon_lam, (lam - 1) / lam
        pw_theory_alpha = -2*np.log(lam*(1-2*w_for_path)) + (1-2*w_for_path*lam*(1-2*alpha/lam-1/lam**2))*epsilon_lam**2
        w_integration_numerical = simps(path[:, 2]+path[:, 3], w_for_path)
        s0=1/lam-1+np.log(lam)
        w_theory_integration=1/lam-1+np.log(lam)-((lam-1)*(1+2*alpha*lam+lam**2)/(4*math.pow(lam, 3)))*epsilon_lam**2
        numerical_correction=w_theory_integration-(1/lam-1+np.log(lam))
        theory_correction=w_integration_numerical-(1/lam-1+np.log(lam))
        pw0=[-2*np.log(lam*(1-2*w)) for w in w_for_path]
        plt.plot(w_for_path, pw_theory_alpha, linewidth=4,
                 linestyle=':', label='correction=' + str(round(theory_correction,5)))


    xlabel('w')
    ylabel('pw')
    title('pw vs w eps='+str(epsilon)+' Lam='+str(beta)+ ' Int='+str(round(numerical_correction,5)))
    plt.legend()
    plt.savefig('pw_vs_w' + '.png', dpi=500)
    plt.show()


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
    plt.savefig('second_order_pu_w'+'.png')
    plt.show()

    plt.plot(u/epsilon_lam,w)
    plt.xlabel('u/eps')
    plt.ylabel('w')
    plt.show()

    if epsilon is not float:
        epsilon_lam, epsilon_mu, lam = epsilon[0], epsilon[1], beta / gamma
        alpha, x0 = epsilon_mu / epsilon_lam, (lam - 1) / lam
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
    plt.savefig('pw_vs_w' + '.png', dpi=500)
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


if __name__=='__main__':
    #Network Parameters
    beta, gamma = 1.5, 1.0

    gamma=1.0
    # beta=[1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,4.5,5.0]

    abserr,relerr = 1.0e-20,1.0e-13
    list_of_epsilons=[(0.1,0.1)]
    # list_of_epsilons=0.1
    sim='bc'

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
    r=8e-07

    epsilon=(0.1,0.0)
    #lin002=0.9999930516412242
    #int_lin_combo001=0.9999658209936237
    # int_lin_combolam5=0.9999658419290037
    # int_lin_comboeps(01,01)=1.0001955976196242
    # int_lin_combo0018e-7=0.9999657791228237
    int_lin_combo=1.0001955976196242
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf,J=eq_hamilton_J(sim, beta, epsilon, t, gamma)
    q_star=[y1_0, y2_0,  p1_star_clancy, p2_star_clancy]
    # man_div_path_and_fine_tuning(np.pi/4-0.785084,r,t,0.9999657791228237,dt,q_star,J,dq_dt_sus_inf,beta)
    # multi_eps_normalized_path(sim,)
    # eq_points_alpha(epsilon, beta, gamma)
    multi_eps_normalized_path(sim, list_of_epsilons, beta, gamma, numpoints, dt, r,int_lin_combo)
