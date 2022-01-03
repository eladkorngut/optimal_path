import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
from pylab import figure, plot, xlabel, grid, legend, title,savefig,ylabel
import numdifftools as ndft
from itertools import cycle
import csv
import pickle
import os
import shooting

from scipy import interpolate



w,u,pw,pu,y1,y2,p1,p2=lambda path:(path[:,0]+path[:,1])/2,lambda path:(path[:,0]-path[:,1])/2,\
                      lambda path:path[:,2]+path[:,3],lambda path:path[:,2]-path[:,3],lambda path:path[:,0],\
                      lambda path:path[:,1],lambda path:path[:,2],lambda path:path[:,3]
y1_div_y2=lambda p:y1(p)/y2(p)
angle,r=0.04239816339744822,1.6384e-08

dfun_dpes = lambda fun,path,path_eps0,eps: (fun(path)-fun(path_eps0))/eps

# correction_s= lambda ds_dq1,dq_deps1,ds_dq2,dq_deps2,eps,path,path_eps0: ds_dq1(path_eps0)*dfun_dpes(dq_deps1,path,path_eps0,eps)+ds_dq2(path_eps0)*dfun_dpes(dq_deps2,path,path_eps0,eps)

correction_s= lambda ds_dq1,dq_deps1,ds_dq2,dq_deps2,eps,path,path_eps0: ds_dq1(path)*dfun_dpes(dq_deps1,path,path_eps0,eps)+ds_dq2(path)*dfun_dpes(dq_deps2,path,path_eps0,eps)

dq_dx = lambda f,x,path,path0: (f(path)-f(path0))/(x(path)-x(path0))

correction_p = lambda f1,f2,x1,x2,path,path0,eps: dq_dx(f1,x1,path,path0)*dfun_dpes(x1,path,path0,eps) + dq_dx(f2,x2,path,path0)*dfun_dpes(x2,path,path0,eps)

approx_const_dpw_deps=lambda eps_mu,lam:2*(1-1/lam)*eps_mu

# phi_const = lambda p,y,y1,y2,mu:np.exp(p)+np.log((mu*(1/2-y)*(y1+y2))/(y))

phi_const = lambda p,y,y1,y2,mu,lam: (-2*y)/(np.exp(p)*(y1 + y2)*(-1 + 2*y)*mu)-lam

s0 =lambda lam:1/lam-1+np.log(lam)

# correction_s_approx= lambda eps_mu,eps_lam,path,path_eps0,lam: w(path_eps0)*approx_const_dpw_deps(eps_mu,lam)+u(path_eps0)*dfun_dpes(pu,path,path_eps0,eps_lam)

correction_s_approx= lambda eps_mu,eps_lam,path,path_eps0,lam:u(path_eps0)*dfun_dpes(pu,path,path_eps0,eps_lam)

action_path=lambda path: simps(path[:, 2], path[:, 0]) + simps(path[:, 3], path[:, 1])

def sim_diff_time(p,e,l,rad,shot):
    path,action_times,action,action_p=[],np.linspace(0.0001,p['stoptime'],1000),[],[]
    for time in action_times:
        y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt, J = shooting.eq_hamilton_J(p['sim'],
                                                                    p['lam'], e, p['time_series'],1.0)
        q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
        path_current_eps =shooting.one_shot(shot, l, q_star, rad, np.linspace(0.0, time, p['numpoints']),
                                            16.0 / (p['numpoints'] - 1),J,shot_dq_dt)
        action.append(simps(path_current_eps[:, 2], path_current_eps[:, 0]) + simps(path_current_eps[:, 3], path_current_eps[:, 1]))
        action_p.append(-(simps(path_current_eps[:, 0], path_current_eps[:, 2]) + simps(path_current_eps[:, 1], path_current_eps[:, 3])))
        path.append(np.array([path_current_eps[:,0][-1],path_current_eps[:,1][-1],path_current_eps[:,2][-1],path_current_eps[:,3][-1]]))
    return np.array(path),np.array(action),np.array(action_p)


def save_action_time_series(v,n):
    path,action_times,action=[],np.linspace(0.0001,v[n]['stoptime'],10),[]
    for p, e, l, rad, shot in zip(v[n]['path'], v[n]['epsilon'], v[n]['lin'], v[n]['radius'],
                                  v[n]['shot_angle']):
        for time in action_times:
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt, J = shooting.eq_hamilton_J(v[n]['sim'],
                                                                        l, e, v[n]['time_series'],1.0)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            path_current_eps =shooting.one_shot(shot, l, q_star, rad, np.linspace(0.0, time, v[n]['numpoints']),
                                                16.0 / (v[n]['numpoints'] - 1),J,shot_dq_dt)
            action.append(simps(path_current_eps[:, 2], path_current_eps[:, 0]) + simps(path_current_eps[:, 3], path_current_eps[:, 1]))
            path.append(np.array([path_current_eps[:,0][-1],path_current_eps[:,1][-1],path_current_eps[:,2][-1],path_current_eps[:,3][-1]]))
    v[n]['action_time_series'],v[n]['path_time_series']=np.array(action),np.array(path)
    pickle.dump(v[n]['action_time_series'],open('action_time_series.pkl','wb'))
    pickle.dump(v[n]['path_time_series'],open('path_time_series.pkl','wb'))
    return v

# (shooting.v_clancy_epslam0(paths[:,0],paths[:,1],var[name]['eps_mu'][0],var[name]['lam'])-shooting.v_clancy_epslam0(var[name]['qstar'][0][0],var[name]['qstar'][0][1],var[name]['eps_mu'][0],var[name]['lam'])))/var[name]['eps_lam'][0]

action_shooting_v_space= lambda p,eps_mu,var,name,q: shooting.v_clancy_epslam0(y1(p),y2(p),eps_mu,
                        var[name]['lam'])-shooting.v_clancy_epslam0(q[0],q[1],eps_mu,var[name]['lam'])

action_shooting_u_space = lambda p,eps_mu,var,name,q: shooting.u_clancy_epslam0(0,0,eps_mu,var[name]['lam']) -\
                shooting.u_clancy_epslam0(p1(p),p2(p),eps_mu,var[name]['lam'])
action_shooting_u_space_epsmu0= lambda p,eps_lam,var,name,q: -shooting.u_clancy_epsmu0(q[0],q[1],eps_lam,var[name]['lam'],1.0) + shooting.u_clancy_epsmu0(y1(p), y2(p), eps_lam, var[name]['lam'],1.0)

def plot_diff_times(name,parameter,xfun,xlabel,ylabel,title,savename,divide_by_eps):
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    # paths_clancy, action_clancy, action_p_clancy = sim_diff_time(var[name], var[name]['epsilon'][4], var[name]['lin'][4],
    #                                         var[name]['radius'][4], var[name]['shot_angle'][4])
    # action_at_epslam0_interplot=interpolate.interp1d(pw(paths_clancy), action_p_clancy, axis=0, fill_value="extrapolate")
    if divide_by_eps is True:
        for p,e,l,rad,shot,eps_mu,q,eps_lam in zip(var[name]['path'],var[name]['epsilon'],var[name]['lin'],var[name]['radius']
                ,var[name]['shot_angle'],var[name]['eps_mu'],var[name]['qstar'],var[name]['eps_lam']):
            if not eps_lam==0:
                paths, action,action_p = sim_diff_time(var[name],e,l,rad,shot)
                # ax.plot(xfun(paths), (action - action_shooting_v_space(paths,eps_mu,var,name,q)) /eps_lam,label='epsilon='+str(e),linewidth=4)
                # ax.plot(xfun(paths), (action_p - action_at_epslam0_interplot(pw(paths)))/eps_lam, label='epsilon=' + str(e),
                #         linewidth=4)
                # ax.plot(xfun(paths), shooting.s1_o1_epslam0(0, eps_mu, var[name]['lam'])-shooting.s1_o1_epslam0(pw(paths), eps_mu, var[name]['lam']), linewidth=4,
                #         linestyle='--')

    else:
        for p,e,l,rad,shot,eps_mu,q,eps_lam in zip(var[name]['path'],var[name]['epsilon'],var[name]['lin'],var[name]['radius']
                ,var[name]['shot_angle'],var[name]['eps_mu'],var[name]['qstar'],var[name]['eps_lam']):
        # for p,e,l,rad,shot,q in zip(var[name]['path'],var[name]['epsilon'],var[name]['lin'],var[name]['radius']
        #         ,var[name]['shot_angle'],var[name]['qstar']):
            paths, action,action_p = sim_diff_time(var[name],e,l,rad,shot)
            # ax.plot(xfun(paths), (action -action_shooting_v_space(paths,eps_mu,var,name,q)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_p,label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_shooting_u_space_epsmu0(paths,e,var,name,q),linewidth=4,linestyle='--')
            # ax.plot(xfun(paths), action_p-action_shooting_u_space_epsmu0(paths,e,var,name,q),linewidth=4,linestyle='--')
            # ax.plot(xfun(paths), action_shooting_v_space(paths,eps_mu,var,name,q),label='epsilon='+str(e),linewidth=4,linestyle='--')
            # ax.plot(xfun(paths), (action_p -action_shooting_u_space(paths,eps_mu,var,name,q)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_p,label='Sim epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_shooting_u_space(paths,eps_mu,var,name,q),label='Theory epsilon='+str(e),linewidth=4,linestyle='--')
            # ax.plot(xfun(paths), (action -action_shooting_u_space(paths,eps_mu,var,name,q)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action - action_clancy,label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_p -action_at_epslam0_interplot(pw(paths)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), shooting.s1_o1_epslam0(pw(paths),eps_mu,var[name]['lam'])*eps_lam,linewidth=4,linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' lam='+str(var[name]['lam']))
    plt.legend()
    plt.tight_layout()
    fig.savefig(savename + '.png', dpi=200)
    plt.show()


def plot_one_path(name,xfun=None,yfun=None,paramter=0,ylabel='',xlabel='',title='',savename='path_plot',divide_by_eps=False):
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    if xfun is False:
        if divide_by_eps is True:
            # paths_clancy, action_clancy, action_p_clancy = sim_diff_time(var[name], var[name]['epsilon'][4], var[name]['lin'][4],
            #                                         var[name]['radius'][4], var[name]['shot_angle'][4])
            for p,e,eps_mu,l,rad,shot,epsilon in zip(var[name]['path'],var[name]['eps_lam'],var[name]['eps_mu'],var[name]['lin'],var[name]['radius']
                ,var[name]['shot_angle'],var[name]['epsilon']):
                if not e==0:
                    # ax.plot(var[name]['time_series'][9500:10000], (yfun(p) - paramter) /e,label='eps_lam='+str(e),linewidth=4)
                    # ax.plot(var[name]['time_series'], (yfun(p) - paramter) /e,label='eps_lam='+str(e),linewidth=4)
                    ax.plot(var[name]['time_series'], phi_const(p1(p), y1(p), y1(p), y2(p), 1 - eps_mu, var[name]['lam'])/e,
                            linewidth=4, label='eps=' + str(epsilon))
                    # ax.plot(var[name]['time_series'], (yfun(p) - p2(paramter)) /e,linewidth=4)
                    # paths, action, action_p = sim_diff_time(var[name], epsilon, l, rad, shot)
                    # ax.plot(var[name]['time_series'], correction_p(p2,p2,y1,y2,p,paramter,e), label='eps_lam=' + str(e), linewidth=4, linestyle='--')
                    # ax.plot(var[name]['time_series'], -correction_s(w,pw,u,pu,e,p,paramter),linewidth=4,linestyle='--')
                    # ax.plot(var[name]['time_series'], -correction_s_approx(eps_mu,e,p,paramter,var[name]['lam']),linewidth=4,linestyle='--')
                    # ax.plot(np.linspace(0.0001,var[name]['stoptime'],1000), (action_p-action_p_clancy)/e,label='eps_lam='+str(e),linewidth=4)
                    # ax.plot(var[name]['time_series'],shooting.p1_linear_approx_dy_deps_epslam_small(y1(p),y2(p),eps_mu,var[name]['lam']),linewidth=4,linestyle='--')
                    # ax.plot(var[name]['time_series'],shooting.p2_linear_approx_dy_deps_epslam_small(y1(var[name_of_file]['path'][4]),y2(var[name_of_file]['path'][4]),eps_mu,var[name]['lam']),linewidth=4,linestyle='--')
                    # ax.plot(var[name]['time_series'],shooting.y1_linear_approx_dy_deps_epslam_small(p1(var[name_of_file]['path'][4]),p2(var[name_of_file]['path'][4]),eps_mu,var[name]['lam']),linewidth=4,linestyle='--')
                    # ax.plot(var[name]['path'][4][:,2]+var[name]['path'][4][:,3], (yfun(p) - paramter) /e,label='eps_lam='+str(e),linewidth=4)
        else:
            for p,e,eps_lam in zip(var[name]['path'],var[name]['eps_mu'],var[name]['eps_lam']):
                # ax.plot(var[name]['time_series'][9500:10000], (yfun(p)[9500:10000] - paramter),label='eps_lam='+str(eps_lam),linewidth=4)
                # ax.plot(var[name]['time_series'], (yfun(p) - paramter),label='eps_lam='+str(eps_lam),linewidth=4)
                ax.plot(var[name]['time_series'],
                        phi_const(p2(p), y2(p), y1(p), y2(p), 1 + e, var[name]['lam']) ,
                        linewidth=4, label='eps=' + str(eps_lam))

                # ax.plot(var[name]['time_series'], phi_const(p2(p), y2(p), y1(p), y2(p), 1 + e,var[name]['lam']), linewidth=4,label='eps_mu='+str(e))
                # ax.plot(var[name]['time_series'], (shooting.y1star_epslam0(e,var[name]['lam'])/shooting.y2star_epslam0(e,var[name]['lam']) - paramter),label='eps_mu='+str(e),linewidth=4,linestyle='--')
                # ax.axhline(y=shooting.y1star_epslam0(e,var[name]['lam'])/shooting.y2star_epslam0(e,var[name]['lam']),  linestyle='--')


    else:
        if divide_by_eps is True:
            for p,eps_mu,eps,eps_lam,q_star in zip(var[name]['path'],var[name]['eps_mu'],var[name]['epsilon'],var[name]['eps_lam'],var[name]['qstar']):
                if not eps_lam==0:
                    # ax.plot(xfun(p), (yfun(p) - paramter) /eps_lam,label='eps='+str(eps),linewidth=4)
                    # w_theory = (shooting.y1_path_clancy_epslam0(p1(p), p2(p), eps_mu,var[name]['lam'])
                    #             + shooting.y2_path_clancy_epslam0(p1(p), p2(p), eps_mu, var[name]['lam'])) / 2
                    # u_theory = (shooting.y1_path_clancy_epslam0(p1(p), p2(p), eps_mu,var[name]['lam'])
                    #             - shooting.y2_path_clancy_epslam0(p1(p), p2(p), eps_mu, var[name]['lam'])) / 2
                    # y1_theory = shooting.dy1_desplam_o0(np.linspace(shooting.y1star_epslam0(eps_mu,var[name]['lam']),0,1000),eps_mu,var[name]['lam'])
                    # y2_theory = shooting.dy2_desplam_o0(np.linspace(shooting.y2star_epslam0(eps_mu,var[name]['lam']),0,1000),eps_mu,var[name]['lam'])
                    # p1_theory = shooting.dp1_depslam_o0(np.linspace(-np.log(var[name]['lam']),0,1000),eps_mu,var[name]['lam'])
                    # p2_theory = shooting.dp2_depslam_o0(np.linspace(-np.log(var[name]['lam']),0,1000),eps_mu,var[name]['lam'])
                    # y_theory_numerical = ((yfun(p)[0]-paramter[0])/(paramter[0]*eps_lam))*paramter
                    # ax.plot(np.linspace(shooting.y1star_epslam0(eps_mu,var[name]['lam']),0,1000), y1_theory,linestyle='--',linewidth=4)
                    # ax.plot(np.linspace(shooting.y2star_epslam0(eps_mu,var[name]['lam']),0,1000), y2_theory,linestyle='--',linewidth=4)
                    # ax.plot(paramter, y_theory_numerical,linestyle='--',linewidth=4)
                    # ax.plot(np.linspace(-np.log(var[name]['lam']),0,1000), p2_theory,linestyle='--',linewidth=4)
                    # ax.plot(xfun(var[name_of_file]['path'][4]),shooting.p2_linear_approx_dy_deps_epslam_small(y1(var[name_of_file]['path'][4]),y2(var[name_of_file]['path'][4]),eps_mu,var[name]['lam']),linewidth=4,linestyle='--')

                    # ax.plot(xfun(p), (yfun(p) - w_theory)/eps_lam, label='eps=' + str(eps), linewidth=4)
                    ax.plot(xfun(p)/eps_mu, (yfun(p) - paramter)/eps_lam, label='eps=' + str(eps), linewidth=4)
                    # ax.plot(xfun(p), (yfun(p) - shooting.y2_path_clancy_epslam0(p1(p), p2(p), eps_mu,var[name]['lam']))/eps_lam, label='eps=' + str(eps), linewidth=4)
                    # ax.plot(xfun(p), (yfun(p) - shooting.p2_path_clancy(y1(p), y2(p), eps_mu,var[name]['lam']))/eps_lam, label='eps=' + str(eps), linewidth=4)
                    # ax.plot(xfun(p), shooting.w_correction_s1_fun_pw(pw(p),eps_mu,var[name]['lam']), linewidth=4,linestyle='--')
                    # ax.plot(xfun(p), (yfun(p) - shooting.p2_delta_o0(y2(p),var[name]['lam']))/(1-eps_mu), label='Sim eps=' + str(eps), linewidth=4)
                    # ax.plot(xfun(p), shooting.p2_delta_mu_o1(xfun(p),eps_lam,var[name]['lam']), label='Theory eps=' + str(eps), linewidth=4,linestyle='--')
                    # ax.plot(xfun(p)/(1-eps_mu), yfun(p) , label='Sim eps=' + str(eps), linewidth=4)
                    # ax.plot(xfun(p)/(1-eps_mu), shooting.p1_delta_mu_o1(xfun(p)/(1-eps_mu),eps_lam,var[name]['lam']) , label='Theory eps=' + str(eps), linewidth=4,linestyle='--')
                    # ax.plot(xfun(p), (yfun(p))/eps_lam, label='eps=' + str(eps), linewidth=4)

        else:
            for p,e,eps in zip(var[name]['path'],var[name]['eps_mu'],var[name]['epsilon']):
                # ax.plot(xfun(p), (yfun(p) -paramter),label='eps='+str(eps),linewidth=4)
                ax.plot(xfun(p), (yfun(p) -xfun(p)/shooting.linear_y1_div_y2(e,xfun(p))),linewidth=4)
                # ax.plot(xfun(p), shooting.linear_y1_div_y2(e,xfun(p)),linewidth=4,linestyle='--')
                # w_theory=(shooting.y1_path_clancy_epslam0(p1(p),p2(p),e,var[name]['lam'])+shooting.y2_path_clancy_epslam0(p1(p),p2(p),e,var[name]['lam']))/2
                # ax.plot(xfun(p), (yfun(p) -w_theory),label='eps='+str(eps),linewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' lam='+str(var[name]['lam']))
    plt.legend()
    plt.tight_layout()
    fig.savefig(savename + '.png', dpi=200)
    plt.show()
    return


def plot_action(name,paramter=0,ylabel='',xlabel='',title='',savename='action_plot',divide_by_eps=False,xaxis=None):
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    def plot_regime_change():
        contact_point = shooting.epslam_crit_regim_lin_sqr(var[name]['eps_mu'], var[name]['lam'])
        sqr_range = np.linspace(0.0, contact_point, 1000)
        lin_range = np.linspace(contact_point, 1.0, 1000)
        # ax.plot(var[name]['eps_lam'], shooting.s2_both_small_correction_to_clancy(np.array(var[name]['eps_lam']),
        #          np.array(var[name]['eps_mu']),var[name]['lam']),label='Theory sqr',linewidth=4,linestyle='-')
        # ax.plot(var[name]['eps_lam'], shooting.s1_epslam_large_minus_clancy(np.array(var[name]['eps_lam']),
        #          np.array(var[name]['eps_mu']),var[name]['lam']),label='Theory lin',linewidth=4,linestyle='--')
        # ax.scatter(shooting.epslam_crit_regim_lin_sqr(var[name]['eps_mu'],var[name]['lam']),
        #     shooting.s1_epslam_large_minus_clancy(shooting.epslam_crit_regim_lin_sqr(var[name]['eps_mu'],var[name]['lam']),
        #     np.array(var[name]['eps_mu']),var[name]['lam']),marker='v',s=100,color='k',zorder=3,label='Regime change')
        ax.plot(sqr_range, shooting.s2_both_small_correction_to_clancy(sqr_range,
                                                                       np.array(var[name]['eps_mu']), var[name]['lam']),
                linewidth=4, linestyle='-', color='r')
        ax.plot(lin_range, shooting.s1_epslam_large_minus_clancy(lin_range,
                                                                 np.array(var[name]['eps_mu']), var[name]['lam']),
                linewidth=4, linestyle='--', color='k')
        ax.scatter(shooting.epslam_crit_regim_lin_sqr(var[name]['eps_mu'], var[name]['lam']),
                   shooting.s1_epslam_large_minus_clancy(
                       shooting.epslam_crit_regim_lin_sqr(var[name]['eps_mu'], var[name]['lam']),
                       np.array(var[name]['eps_mu']), var[name]['lam']), marker='v', s=100, color='m', zorder=3,
                   label='Regime change')
        if divide_by_eps is True:
            ax.plot(xaxis, (var[name]['action_paths'] - paramter(var[name]['eps_mu'],var[name]['lam'],1.0)) /var[name]['eps_lam'],label='Sim',linewidth=4,
                    linestyle='None',markersize=10, Marker='o',)
        else:
            ax.plot(xaxis, (var[name]['action_paths'] - paramter(var[name]['eps_mu'],var[name]['lam'],1.0)),label='Sim',linewidth=4,linestyle='None'
                    ,markersize=10, Marker='o')
        ax.autoscale(False)  # To avoid that the scatter changes limits
    def plot_multi_epslam_const_epsmu_change():
        for name_current in name:
            lam, x0 = var[name_current]['lam'], 1 - 1 / var[name_current]['lam']
            eps_mu_regime_change = (-1 + lam - lam ** 2 + lam ** 3 - 2 * lam * np.log(lam) - 2 * var[name_current][
                'eps_lam'] + 4 * lam * var[name_current]['eps_lam']- 2 * lam ** 2 * var[name_current]['eps_lam']) / (
               1 - 3 * lam + lam ** 2 + lam ** 3 - 2 * lam * np.log(lam))
            epsmu1_epslam0_norm = lambda lam, eps_lam, eps_mu: ((((-1 + lam) * (1 + lam ** 2) - 2 * lam * np.log(
                lam)) * eps_lam * (-1 + eps_mu)) / (4 * lam ** 2)) / eps_lam
            epsmu0_epslam0_norm = lambda lam, eps_lam, eps_mu: -(1 / 2) * x0 ** 2 * (
                        eps_mu * eps_lam + eps_lam ** 2) / eps_lam
            epsmu1_epslam0 = lambda lam, eps_lam, eps_mu: ((((-1 + lam) * (1 + lam ** 2) - 2 * lam * np.log(
                lam)) * eps_lam * (-1 + eps_mu)) / (4 * lam ** 2))
            epsmu0_epslam0 = lambda lam, eps_lam, eps_mu: -(1 / 2) * x0 ** 2 * (
                        eps_mu * eps_lam + eps_lam ** 2)
            # eps_lam_theory = np.linspace(min(var[name]['eps_lam']),max(var[name]['eps_lam']),1000)
            eps_lam_theory = min(var[name_current]['eps_lam'])
            eps_mu_theory_low = np.linspace(0.0, eps_mu_regime_change, 1000)
            eps_mu_theory_high = np.linspace(eps_mu_regime_change, 1.0, 1000)

            if divide_by_eps is True:
                # ax.plot(xaxis, (var[name]['action_paths'] - paramter) /var[name]['eps_lam'],
                #         label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o',)
                # ax.plot(xaxis, (var[name_current]['action_paths'] - shooting.action_clancy(xaxis,lam,1.0)) /var[name_current]['eps_lam'],
                #         label='Sim eps_lam='+str(eps_lam_theory),linewidth=4,linestyle='None',markersize=10, Marker='o',)
                ax.plot(var[name_current]['eps_mu'], (var[name_current]['action_paths'] - shooting.action_clancy(
                    var[name_current]['eps_mu'], lam,1.0)) / var[name_current]['eps_lam'],label='Sim eps_lam=' +
                    str(eps_lam_theory), linewidth=4, linestyle='None', markersize=10,Marker='o')
            else:
                # ax.plot(xaxis, (var[name]['action_paths'] - paramter),label='Sim',
                #         linewidth=4,linestyle='None',markersize=10, Marker='o',)
                ax.plot(var[name_current]['eps_mu'], (var[name_current]['action_paths'] - shooting.action_clancy(var[name_current]['eps_mu'], lam, 1.0)),
                        label='Sim'+' eps='+str(eps_lam_theory),linewidth=4, linestyle='None', markersize=10, Marker='o')
            if divide_by_eps is False:
                ax.plot(eps_mu_theory_low, epsmu0_epslam0(lam, eps_lam_theory, eps_mu_theory_low), linewidth=4,
                        linestyle='-', color='r')
                ax.plot(eps_mu_theory_high, epsmu1_epslam0(lam, eps_lam_theory, eps_mu_theory_high), linewidth=4,
                        linestyle='--', color='k')
            else:
                ax.plot(eps_mu_theory_low, epsmu0_epslam0_norm(lam, eps_lam_theory, eps_mu_theory_low), linewidth=4,
                        linestyle='-', color='r')
                ax.plot(eps_mu_theory_high, epsmu1_epslam0_norm(lam, eps_lam_theory, eps_mu_theory_high), linewidth=4,
                        linestyle='--', color='k')

            # ax.plot(np.linspace(0.01,0.99,1000),shooting.action_o1_epsmu(eps_lam_theory,np.linspace(0.001,0.99,1000),lam),color='tab:orange',linewidth=4,linestyle=':')

    def plot_diff_lam():
        lam_theory = np.linspace(min(var[name]['lam']),max(var[name]['lam']),1000)
        if divide_by_eps is True:
            # ax.plot(var[name]['lam'], (var[name]['action_paths'] - shooting.action_clancy(var[name]['eps_mu'],var[name]['lam'],1.0)) /var[name]['eps_lam'],
            #         label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o',)
            ax.plot(var[name]['lam'], (var[name]['action_paths']-s0(var[name]['lam'])/2)/var[name]['eps_lam'],
                    label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o',)
        else:
            # ax.plot(var[name]['lam'], (var[name]['action_paths'] - shooting.action_clancy(var[name]['eps_mu'],var[name]['lam'],1.0)),label='Sim',
            #         linewidth=4,linestyle='None',markersize=10, Marker='o')
            # ax.plot(var[name]['lam'], var[name]['action_paths']-s0(var[name]['lam'])/2,label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o')
            # ax.plot(var[name]['lam'],var[name]['action_paths']-s0(var[name]['lam']),label='Sim',linewidth=4,linestyle='None'
            #         ,markersize=10, Marker='o')
            # ax.plot(var[name]['lam'],var[name]['action_paths']-s0(var[name]['lam']) - shooting.s2_both_small
            # (var[name]['eps_lam'],var[name]['eps_mu'],var[name]['lam']),label='ME',linewidth=4,linestyle='None',markersize=10, Marker='o')
            # ax.plot(var[name]['lam'],var[name]['action_paths']-s0(var[name]['lam']),label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o')
            # ax.plot(var[name]['lam'],np.abs((var[name]['action_paths']-s0(var[name]['lam']) - shooting.action_miki_jason_correction
            # (var[name]['eps_lam'],var[name]['lam']))/(var[name]['action_paths']-s0(var[name]['lam']))),label='MJ',linewidth=4,linestyle='None',markersize=10, Marker='v')
            ax.plot(var[name]['lam'],np.abs((var[name]['action_paths']-s0(var[name]['lam']) - shooting.s2_both_small
            (var[name]['eps_lam'],var[name]['eps_mu'],var[name]['lam']))/(var[name]['action_paths']-s0(var[name]['lam']))),label='ME',linewidth=4,linestyle='None',markersize=10, Marker='v')
            # ax.plot(var[name]['lam'],var[name]['action_paths']-s0(var[name]['lam']) ,label='ME',linewidth=4,linestyle='None',markersize=10, Marker='v')
            # ax.plot(var[name]['lam'],var[name]['action_paths']-shooting.action_clancy(var[name]['eps_mu'],var[name]['lam'],1.0) ,label='ME',linewidth=4,linestyle='None',markersize=10, Marker='v')
        # ax.plot(lam_theory,shooting.s1_epslam_large(var[name]['eps_lam'],var[name]['eps_mu'],lam_theory), linewidth=4,
        #         linestyle='-', color='r',label='Theory')
        # ax.plot(lam_theory,shooting.s2_both_small(var[name]['eps_lam'],var[name]['eps_mu'],lam_theory), linewidth=4,
        #         linestyle='-', color='r',label='ME')
        # ax.plot(lam_theory,shooting.action_miki_jason_correction(var[name]['eps_lam'],lam_theory), linewidth=4,
        #         linestyle='--', color='k',label='MJ')
        # ax.plot(lam_theory, epsmu1_epslam0(lam_theory, var[name]['eps_lam'], var[name]['eps_mu']), linewidth=4,
        #         linestyle='--', color='k')

    def plot_diff_lam_multi():
        for name_current in name:
            lam_theory = np.linspace(min(var[name_current]['lam']), max(var[name_current]['lam']), 1000)
            if divide_by_eps is True:
                # ax.plot(var[name_current]['lam'], (var[name_current]['action_paths'] - shooting.action_clancy(var[name_current]['eps_mu'],var[name]['lam'],1.0)) /var[name]['eps_lam'],
                #         label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o',)
                ax.plot(var[name_current]['lam'], (var[name_current]['action_paths'] - s0(var[name_current]['lam']) / 2) / var[name]['eps_lam'],
                        label='Sim', linewidth=4, linestyle='None', markersize=10, Marker='o', )
            else:
                # ax.plot(var[name_current]['lam'], (var[name_current]['action_paths'] - shooting.action_clancy(var[name_current]['eps_mu'],var[name_current]['lam'],1.0)),label='Sim',
                #         linewidth=4,linestyle='None',markersize=10, Marker='o')
                # ax.plot(var[name_current]['lam'], var[name_current]['action_paths']-s0(var[name_current]['lam'])/2,label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o')
                # ax.plot(var[name_current]['lam'],var[name_current]['action_paths']-s0(var[name_current]['lam']),label='Sim',linewidth=4,linestyle='None'
                #         ,markersize=10, Marker='o')
                # ax.plot(var[name_current]['lam'],var[name_current]['action_paths']-s0(var[name_current]['lam']) - shooting.s2_both_small
                # (var[name_current]['eps_lam'],var[name_current]['eps_mu'],var[name_current]['lam']),label='ME',linewidth=4,linestyle='None',markersize=10, Marker='o')
                # ax.plot(var[name_current]['lam'],var[name_current]['action_paths']-s0(var[name_current]['lam']),label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o')
                # ax.plot(var[name_current]['lam'],np.abs((var[name_current]['action_paths']-s0(var[name_current]['lam']) - shooting.action_miki_jason_correction
                # (var[name_current]['eps_lam'],var[name_current]['lam']))/(var[name_current]['action_paths']-s0(var[name_current]['lam']))),label='MJ',linewidth=4,linestyle='None',markersize=10, Marker='v')
                # ax.plot(var[name_current]['lam'],
                #         np.abs((var[name_current]['action_paths'] - s0(var[name_current]['lam']) - shooting.s2_both_small
                #         (var[name_current]['eps_lam'], var[name_current]['eps_mu'], var[name_current]['lam'])) / (
                #                            var[name_current]['action_paths'] - s0(var[name_current]['lam']))), linewidth=4,
                #         linestyle='None', markersize=10, Marker='v')
                # ax.plot(var[name_current]['lam'],-(var[name_current]['action_paths'] - s0(var[name_current]['lam'])), linewidth=4,
                #         linestyle='None', markersize=10, Marker='o')
                ax.plot(var[name_current]['lam'],(var[name_current]['action_paths'] - s0(var[name_current]['lam']))/shooting.s2_both_small(var[name_current]['eps_lam'],var[name_current]['eps_mu'],var[name_current]['lam']), linewidth=4,
                        linestyle='None', markersize=10, Marker='o',label='ME eps='+str(var[name_current]['eps_lam']))
                # ax.plot(var[name_current]['lam'],(var[name_current]['action_paths'] - s0(var[name_current]['lam']))/np.array(shooting.action_miki_jason_correction(var[name_current]['eps_lam'],np.array(var[name_current]['lam']))), linewidth=4,
                #         linestyle='None', markersize=10, Marker='v',label='MJ eps='+str(var[name_current]['eps_lam']))
                # ax.plot(var[name]['lam'],var[name]['action_paths']-s0(var[name]['lam']) ,label='ME',linewidth=4,linestyle='None',markersize=10, Marker='v')
                # ax.plot(var[name]['lam'],var[name]['action_paths']-shooting.action_clancy(var[name]['eps_mu'],var[name]['lam'],1.0) ,label='ME',linewidth=4,linestyle='None',markersize=10, Marker='v')
            # ax.plot(lam_theory,shooting.s1_epslam_large(var[name]['eps_lam'],var[name]['eps_mu'],lam_theory), linewidth=4,
            #         linestyle='-', color='r',label='Theory')
            # ax.plot(lam_theory,-shooting.s2_both_small(var[name_current]['eps_lam'],var[name_current]['eps_mu'],lam_theory), linewidth=4,
            #         linestyle='-',label='Theory eps_lam='+str(var[name_current]['eps_lam']))
            # ax.plot(lam_theory,shooting.action_miki_jason_correction(var[name]['eps_lam'],lam_theory), linewidth=4,
            #         linestyle='--', color='k',label='MJ')
            # ax.plot(lam_theory, epsmu1_epslam0(lam_theory, var[name]['eps_lam'], var[name]['eps_mu']), linewidth=4,
            #         linestyle='--', color='k')

    # plot_diff_lam()
    plot_diff_lam_multi()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title+ ', lam='+str(var[name]['lam']))
    # plt.title(title+ ', eps_lam='+str(var[name]['eps_lam'])+ ', eps_mu='+str(var[name]['eps_mu']))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(savename + '.png', dpi=200)
    plt.show()
    return

def import_folder():
    # import a single folder with all of it data files (pickle) and return them as variables
    file_list,variables=['guessed_paths.pkl', 'lin_combo.pkl', 'qstar.pkl', 'gamma.pkl', 'numpoints.pkl', 'sim.pkl',
                         'stoptime.pkl', 'epsilon_matrix.pkl', 'time_series.pkl', 'action_paths.pkl', 'beta.pkl'],[]
    for f in file_list:
        with open(f,'rb') as pickle_file:
            variables.append(pickle.load(pickle_file))
        pickle_file.close()
    return variables


def import_folder_dict(dict):
    # import a single folder with all of it data files (pickle) and return them as variables
    with open('guessed_paths.pkl', 'rb') as pickle_file:
        dict['path']=(pickle.load(pickle_file)[0])
    pickle_file.close()
    with open('lin_combo.pkl', 'rb') as pickle_file:
        dict['lin']=pickle.load(pickle_file)[0]
    pickle_file.close()
    with open('qstar.pkl', 'rb') as pickle_file:
        dict['qstar']=pickle.load(pickle_file)[0]
    pickle_file.close()
    with open('gamma.pkl', 'rb') as pickle_file:
        gamma=pickle.load(pickle_file)
    pickle_file.close()
    with open('beta.pkl', 'rb') as pickle_file:
        beta = pickle.load(pickle_file)
        beta=np.array(beta) if type(beta) is list else beta
        dict['lam']=beta/gamma
    pickle_file.close()
    with open('numpoints.pkl', 'rb') as pickle_file:
        dict['numpoints']=pickle.load(pickle_file)
    pickle_file.close()
    with open('sim.pkl', 'rb') as pickle_file:
        dict['sim']=pickle.load(pickle_file)[0]
    pickle_file.close()
    with open('stoptime.pkl', 'rb') as pickle_file:
        dict['stoptime']=pickle.load(pickle_file)[0]
    pickle_file.close()
    with open('epsilon_matrix.pkl', 'rb') as pickle_file:
        if type(dict['lam']) is np.ndarray:
            dict['epsilon'] = pickle.load(pickle_file)
            dict['eps_lam'] = dict['epsilon'][0]
            dict['eps_mu'] = dict['epsilon'][1]
        else:
            epsilon_matrix=pickle.load(pickle_file)[0]
            dict['epsilon'] = epsilon_matrix
            if type(epsilon_matrix[0]) is float:
                dict['eps_lam'] = epsilon_matrix
                dict['eps_mu'] = 0
            else:
                eps_lam,eps_mu=[],[]
                for eps in epsilon_matrix:
                    eps_lam.append(eps[0])
                    eps_mu.append(eps[1])
                dict['eps_lam'] = np.array(eps_lam)
                dict['eps_mu'] = np.array(eps_mu)
    pickle_file.close()
    with open('action_paths.pkl', 'rb') as pickle_file:
        dict['action_paths']=np.array(pickle.load(pickle_file)[0])
    pickle_file.close()
    with open('radius.pkl', 'rb') as pickle_file:
        dict['radius']=np.array(pickle.load(pickle_file)[0])
    pickle_file.close()
    with open('shot_angle.pkl', 'rb') as pickle_file:
        dict['shot_angle']=np.array(pickle.load(pickle_file)[0])
    pickle_file.close()
    # with open('partial_paths.pkl', 'rb') as pickle_file:
    #     dict['partial_paths']=np.array(pickle.load(pickle_file)[0])
    # pickle_file.close()
    # with open('partial_action.pkl', 'rb') as pickle_file:
    #     dict['partial_action']=np.array(pickle.load(pickle_file)[0])
    # pickle_file.close()
    dict['time_series']=np.linspace(0.0,dict['stoptime'],dict['numpoints'])
    return dict


def import_all_folders_from_data():
    # import all the data from the files
    os.chdir('/home/elad/optimal_path_numeric/Data')
    # var=[]
    var={}
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            os.chdir(name)
            # var.append(import_folder())
            var[name]=import_folder_dict({})
            # var[name]=save_action_time_series(var,name)
            os.chdir('..')
    return var,dirs


if __name__=='__main__':
    var,filenames= import_all_folders_from_data()
    # name_of_file='eps_mu05_epslam_change_small_stoptime20_with_rad_angle'
    # name_of_file='epslam006_epsmu01_diff_lam'
    # name_of_file=['epslam005_epsmu_change_1_to_0_lam16_stoptime20_linspace20','epslam01_epsmu_change_0_to_1_lam16_stoptime20_linespace_40','epslam015_epsmu_change_1_to_0_lam16_stoptime20_linspace20']
    name_of_file=['epslam002_epsmu002_diff_lam','epslam004_epsmu004_diff_lam','epslam006_epsmu006_diff_lam','epslam012_epsmu012_diff_lam','epslam014_epsmu014_diff_lam']

    # theory=shooting.y1_path_clancy(var[name_of_file]['path'][4][:,2],var[name_of_file]['path'][4][:,3],var[name_of_file]['eps_mu'][4],var[name_of_file]['lam'])
    # plot_one_path(name_of_file,u,pu,pu(var[name_of_file]['path'][4]),'(p1-p1(0))/eps_lam', 'p1', '(p1-p1(0))/eps_lam vs p1','dp1_norm_v_p1',True)
    # plot_one_path(name_of_file,y1,y1,y1(var[name_of_file]['path'][4]),'(p2-p2(0))/eps_lam', 'p2', '(p2-p2(0))/eps_lam vs p2','dp2_v_p2',True)
    # plot_one_path(name_of_file,u,pu,0,'w', 'time', 'w vs time','w_v_time_non_norm',True)
    # plot_one_path(name_of_file,u,pu,0,'pu/eps', 'u/eps', 'pu/eps_lam vs u/eps_mu','temp',True)
    # plot_one_path(name_of_file,False,y1,y1(var[name_of_file]['path'][4]),'(y1-y1(0))/eps_lam', 'time', '(y1-y1(0))/eps_lam vs y1','dy1_v_time_epslam05_with_theory',True)
    # plot_one_path(name_of_file,False,p2,var[name_of_file]['path'][4],'(p2-p2(0))/eps_lam', 'time', '(p2-p2(0))/eps_lam vs time,','dp2_v_time_epslam05',True)
    # plot_one_path(name_of_file,False,action_path,action_path(var[name_of_file]['path'][0]),'', '', '','temp',True)
    # plot_one_path(name_of_file,False,p1,0,'phi(p1,y1)/eps_lam', 'time', 'phi(p1,y1) vs time,','phi1_v_time_epsmu05_epslam_changes',False)
    # plot_one_path(name_of_file,False,p2,0,'phi(p2,y2)', 'time', 'phi(p2,y2) vs time,','phi_p2_y2_v_time_non_norm',False)
    # plot_action(name_of_file,action_path(var[name_of_file]['path'][0]), 'action', 'eps_lam', 'Action vs eps_lam', 'action_plot_range_change_epsmu02_lam16_stoptime20', False)
    plot_action(name_of_file,0, 'S/S_theory', 'lam', 'S/S_theory vs lam','s_div_theory_s_lam_only_me', False,0)
    # plot_diff_times(name_of_file, 0, y1, 'y1', 'action', 'Action vs y1', 'action_v_y1_sub', False)
    # plot_diff_times(name_of_file, 0, y1, 'y1', 'S-S(0)', 'S-S(0) vs y1', 'action_v_y1_epslam_changes_clancy_theory_error', False)
    # plot_one_path(name_of_file,False,action_path,action_path(var[name_of_file]['path'][0]),'', '', '','temp',True)
    print('this no love song')