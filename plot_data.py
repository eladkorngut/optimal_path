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

angle,r=0.04239816339744822,1.6384e-08


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


def plot_diff_times(name,parameter,xfun,xlabel,ylabel,title,savename,divide_by_eps):
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    paths_clancy, action_clancy, action_p_clancy = sim_diff_time(var[name], var[name]['epsilon'][4], var[name]['lin'][4],
                                            var[name]['radius'][4], var[name]['shot_angle'][4])
    action_at_epslam0_interplot=interpolate.interp1d(pw(paths_clancy), action_p_clancy, axis=0, fill_value="extrapolate")
    if divide_by_eps is True:
        for p,e,l,rad,shot,eps_mu,q,eps_lam in zip(var[name]['path'],var[name]['epsilon'],var[name]['lin'],var[name]['radius']
                ,var[name]['shot_angle'],var[name]['eps_mu'],var[name]['qstar'],var[name]['eps_lam']):
            if not eps_lam==0:
                paths, action,action_p = sim_diff_time(var[name],e,l,rad,shot)
                # ax.plot(xfun(paths), (action - action_shooting_v_space(paths,eps_mu,var,name,q)) /eps_lam,label='epsilon='+str(e),linewidth=4)
                ax.plot(xfun(paths), (action_p - action_at_epslam0_interplot(pw(paths)))/eps_lam, label='epsilon=' + str(e),
                        linewidth=4)
                ax.plot(xfun(paths), shooting.s1_o1_epslam0(0, eps_mu, var[name]['lam'])-shooting.s1_o1_epslam0(pw(paths), eps_mu, var[name]['lam']), linewidth=4,
                        linestyle='--')

    else:
        for p,e,l,rad,shot,eps_mu,q,eps_lam in zip(var[name]['path'],var[name]['epsilon'],var[name]['lin'],var[name]['radius']
                ,var[name]['shot_angle'],var[name]['eps_mu'],var[name]['qstar'],var[name]['eps_lam']):
            paths, action,action_p = sim_diff_time(var[name],e,l,rad,shot)
            # ax.plot(xfun(paths), (action -action_shooting_v_space(paths,eps_mu,var,name,q)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), (action_p -action_shooting_u_space(paths,eps_mu,var,name,q)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_p,label='Sim epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action_shooting_u_space(paths,eps_mu,var,name,q),label='Theory epsilon='+str(e),linewidth=4,linestyle='--')
            # ax.plot(xfun(paths), (action -action_shooting_u_space(paths,eps_mu,var,name,q)),label='epsilon='+str(e),linewidth=4)
            # ax.plot(xfun(paths), action - action_clancy,label='epsilon='+str(e),linewidth=4)
            ax.plot(xfun(paths), action_p -action_at_epslam0_interplot(pw(paths)),label='epsilon='+str(e),linewidth=4)
            ax.plot(xfun(paths), shooting.s1_o1_epslam0(pw(paths),eps_mu,var[name]['lam'])*eps_lam,linewidth=4,linestyle='--')

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
            for p,e in zip(var[name]['path'],var[name]['eps_lam']):
                if not e==0:
                    ax.plot(var[name]['time_series'], (yfun(p) - paramter) /e,label='eps_lam='+str(e),linewidth=4)
                    # ax.plot(var[name]['path'][4][:,2]+var[name]['path'][4][:,3], (yfun(p) - paramter) /e,label='eps_lam='+str(e),linewidth=4)
        else:
            for p,e in zip(var[name]['path'],var[name]['eps_lam']):
                ax.plot(var[name]['time_series'], (yfun(p) - paramter),label='eps_lam='+str(e),linewidth=4)
    else:
        if divide_by_eps is True:
            for p,e in zip(var[name]['path'],var[name]['eps_lam']):
                if not e==0:
                    ax.plot(xfun(p), (yfun(p) - paramter) /e,label='eps_lam='+str(e),linewidth=4)
        else:
            for p,e in zip(var[name]['path'],var[name]['eps_lam']):
                ax.plot(xfun(p), (yfun(p) -paramter),label='eps_lam='+str(e),linewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' lam='+str(var[name]['lam']))
    plt.legend()
    plt.tight_layout()
    fig.savefig(savename + '.png', dpi=200)
    plt.show()
    return


def plot_action(name,paramter=0,ylabel='',xlabel='',title='',savename='action_plot',divide_by_eps=False):
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    if divide_by_eps is True:
        ax.plot(var[name]['eps_lam'], (var[name]['action_paths'] - paramter) /var[name]['eps_mu'],
                label='Sim',linewidth=4,linestyle='None',markersize=10, Marker='o',)
    else:
        ax.plot(var[name]['eps_lam'], (var[name]['action_paths'] - paramter),label='Sim',
                linewidth=4,linestyle='None',markersize=10, Marker='o',)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' lam='+str(var[name]['lam']))
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
    name_of_file='eps_mu02_epslam_small_change_stoptime20'
    # theory=shooting.y1_path_clancy(var[name_of_file]['path'][4][:,2],var[name_of_file]['path'][4][:,3],var[name_of_file]['eps_mu'][4],var[name_of_file]['lam'])
    # plot_one_path(name_of_file,u,pu,pu(var[name_of_file]['path'][4]),'(p1-p1(0))/eps_lam', 'p1', '(p1-p1(0))/eps_lam vs p1','dp1_norm_v_p1',True)
    # plot_one_path(name_of_file,y1,y2,0,'(y2)', 'y1', 'y2 vs y1','dw_norm_v_w',False)
    # plot_action(name_of_file,0, 'action', 'eps_lam', 'Action vs eps_lam', 'action_plot', False)
    # plot_diff_times(name_of_file, 0, y1, 'y1', 'action', 'Action vs y1', 'action_v_y1_sub', False)
    plot_diff_times(name_of_file, 0, pw, 'pw', '(S-S(0))/eps_mu', 'S1 (action first order) vs pw', 'action_v_pw_epsmu02', True)
    print('this no love song')