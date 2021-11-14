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
import pandas as pd

w,u,pw,pu,y1,y2,p1,p2=lambda path:(path[:,0]+path[:,1])/2,lambda path:(path[:,0]-path[:,1])/2,\
                      lambda path:path[:,2]+path[:,3],lambda path:path[:,2]-path[:,3],lambda path:path[:,0],\
                      lambda path:path[:,1],lambda path:path[:,2],lambda path:path[:,3]

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
        dict['lin']=pickle.load(pickle_file)[0][0]
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
        if type(epsilon_matrix[0]) is float:
            dict['epsilon'] = epsilon_matrix
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
            os.chdir('..')
    return var,dirs


if __name__=='__main__':
    var,filenames= import_all_folders_from_data()
    name_of_file='eps_mu05_epslam_change_small_stoptime20'
    # theory=shooting.y1_path_clancy(var[name_of_file]['path'][4][:,2],var[name_of_file]['path'][4][:,3],var[name_of_file]['eps_mu'][4],var[name_of_file]['lam'])
    # plot_one_path(name_of_file,u,pu,pu(var[name_of_file]['path'][4]),'(p1-p1(0))/eps_lam', 'p1', '(p1-p1(0))/eps_lam vs p1','dp1_norm_v_p1',True)
    plot_one_path(name_of_file,y1,y2,0,'(y2)', 'y1', 'y2 vs y1','dw_norm_v_w',False)
    plot_action(name_of_file,0, 'action', 'eps_lam', 'Action vs eps_lam', 'action_plot', False)
    print('this no love song')