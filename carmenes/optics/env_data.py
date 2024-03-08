import glob
from astropy.time import Time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import vis_spectrometer
from . import echelle_orders
from . import parameters


def get_CCD_T_vis(mjd):
    # path_env_data = '/home/marcelo/Documents/ramses/vis/env_data/'
    path_env_data = 'env_data/'
    ccd_temp_ts01_file = 'VIS-CR-Ts01.dat'
    ccd_temp_ts02_file = 'VIS-CR-Ts02.dat'
    ccd_temp_data_vis_ts01 = pd.read_csv(path_env_data + ccd_temp_ts01_file, sep=',')
    ccd_temp_data_vis_ts02 = pd.read_csv(path_env_data + ccd_temp_ts02_file, sep=',')

    temp_ts01 = ccd_temp_data_vis_ts01.loc[ccd_temp_data_vis_ts01['mjd'] < mjd + .02]
    temp_ts01 = temp_ts01.loc[temp_ts01['mjd'] > mjd - .02]

    temp_ts02 = ccd_temp_data_vis_ts02.loc[ccd_temp_data_vis_ts02['mjd'] < mjd + .02]
    temp_ts02 = temp_ts02.loc[temp_ts02['mjd'] > mjd - .02]

    avg_temp_ts01 = np.average(temp_ts01[' temp'].values)
    avg_temp_ts02 = np.average(temp_ts02[' temp'].values)

    ccd_temp = (avg_temp_ts01 + avg_temp_ts02)/2
    ccd_temp = ccd_temp - 273.15

    return ccd_temp

    #env_data_vis = env_data_vis.loc[env_data_vis['mjd'] > t_mjd - 25.5]
    #env_data_vis = env_data_vis.loc[(env_data_vis[' temp'] > 278) & (env_data_vis[' temp'] < 285)]

    #max_temp = max(env_data_vis[' temp'].values)
    #min_temp = min(env_data_vis[' temp'].values)
    #delta_T = np.abs(max_temp - min_temp)

    #temp_out = np.average(env_data_vis[' temp'].values)
    #temp_out = temp_out - 273.15
    # print(temp_out)
    #return temp_out


def get_temps():
    mjdfile = open('data/ws/mjd.dat','r')
    line = mjdfile.readline()
    mjd = line.split('\'')
    mjd = float(mjd[1])
    path_env_data = 'env_data/'
    temps = [get_CCD_T_vis(mjd)]
    for i in range(9):
        if i != 0:
            env_data_vis_file = 'VIS-IS-Ts0' + str(i) + '.dat'
            env_data_vis = pd.read_csv(path_env_data + env_data_vis_file, sep=',')
            t_mjd = mjd
            env_data_vis = env_data_vis.loc[env_data_vis['mjd'] < t_mjd + 0.02]
            env_data_vis = env_data_vis.loc[env_data_vis['mjd'] > t_mjd - 0.02]
            temp_out = np.average(env_data_vis[' temp'].values)
            temp_out = temp_out - 273.15
            #print(i, temp_out)
            temps.append(temp_out)

    temps = np.array(temps)
    return temps


def get_T_at_ws(date):
    path_data = 'data/vis_ws_timeseries/'+date+'/'
    mjdfile = open(path_data + 'mjd.dat', 'r')
    line = mjdfile.readline()
    mjd = line.split('\'')
    mjd = float(mjd[1])
    path_env_data = 'env_data/'
    temps = [get_CCD_T_vis(mjd)]
    for i in range(9):
        if i != 0:
            env_data_vis_file = 'VIS-IS-Ts0' + str(i) + '.dat'
            env_data_vis = pd.read_csv(path_env_data + env_data_vis_file, sep=',')
            t_mjd = mjd
            env_data_vis = env_data_vis.loc[env_data_vis['mjd'] < t_mjd + 0.02]
            env_data_vis = env_data_vis.loc[env_data_vis['mjd'] > t_mjd - 0.02]
            temp_out = np.average(env_data_vis[' temp'].values)
            temp_out = temp_out - 273.15
            # print(i, temp_out)
            temps.append(temp_out)

    temps = np.array(temps)
    return temps

    #env_data_vis = env_data_vis.loc[env_data_vis['mjd'] > t_mjd - 25.5]
    #env_data_vis = env_data_vis.loc[(env_data_vis[' temp'] > 278) & (env_data_vis[' temp'] < 285)]

    #max_temp = max(env_data_vis[' temp'].values)
    #min_temp = min(env_data_vis[' temp'].values)
    #delta_T = np.abs(max_temp - min_temp)

    #temp_out = np.average(env_data_vis[' temp'].values)
    #temp_out = temp_out - 273.15
    # print(temp_out)
    #return temp_out


def get_P_at_ws(date):
    path_data = 'data/vis_ws_timeseries/'+date+'/'
    mjdfile = open(path_data + 'mjd.dat', 'r')
    line = mjdfile.readline()
    mjd = line.split('\'')
    mjd = float(mjd[1])
    path_env_data = 'env_data/'
    p_vt_s1 = pd.read_csv(path_env_data + 'VIS-VT-S1.dat', sep=',')
    t_mjd = mjd
    p_date = p_vt_s1.loc[p_vt_s1['mjd'] < t_mjd + 0.02]
    p_date = p_date.loc[p_date['mjd'] > t_mjd - 0.02]
    pdate = np.mean(p_date[' p'].values)
    return pdate


def get_temps_bjd(bjd):

    path_env_data = 'env_data/'
    temps = []
    temps.append(get_CCD_T_vis(bjd))
    for i in range(9):
        if i != 0:
            env_data_vis_file = 'VIS-IS-Ts0' + str(i) + '.dat'
            env_data_vis = pd.read_csv(path_env_data + env_data_vis_file, sep=',')
            t_mjd = bjd - 2400000.5
            env_data_vis = env_data_vis.loc[env_data_vis['mjd'] < t_mjd + 0.02]
            env_data_vis = env_data_vis.loc[env_data_vis['mjd'] > t_mjd - 0.02]
            temp_out = np.average(env_data_vis[' temp'].values)
            temp_out = temp_out - 273.15
            #print(i, temp_out)
            temps.append(temp_out)

    temps = np.array(temps)

    return temps


def get_p_bjd(bjd):
    path_env_data = 'env_data/'
    p_vt_s1 = pd.read_csv(path_env_data+'VIS-VT-S1.dat', sep=',')
    t_mjd = bjd - 2400000.5
    p_date = p_vt_s1.loc[p_vt_s1['mjd'] < t_mjd + 0.02]
    p_date = p_date.loc[p_date['mjd'] > t_mjd - 0.02]
    pout = np.mean(p_date[' p'].values)

    return pout


def get_p():
    mjdfile = open('data/ws/mjd.dat', 'r')
    line = mjdfile.readline()
    mjd = line.split('\'')
    mjd = float(mjd[1])
    path_env_data = 'env_data/'
    p_vt_s1 = pd.read_csv(path_env_data + 'VIS-VT-S1.dat', sep=',')
    t_mjd = mjd
    p_date = p_vt_s1.loc[p_vt_s1['mjd'] < t_mjd + 0.02]
    p_date = p_date.loc[p_date['mjd'] > t_mjd - 0.02]
    pdate = np.mean(p_date[' p'].values)
    return pdate


def get_t_mjd(mjd):
    datadir = 'env_data/'
    tempout = []
    for i in range(8):
        sensorno = i + 1
        temp_data = pd.read_csv(datadir + 'VIS-IS-Ts0' + str(sensorno) + '.dat', sep=',')
        temp_data = temp_data.loc[temp_data['mjd'] < float(mjd) + 0.025]
        temp_data = temp_data.loc[temp_data['mjd'] > float(mjd) - 0.025]
        temp_out = np.average(temp_data[' temp'].values)
        temp_out = temp_out - 273.15
        tempout.append(temp_out)

    tempout = np.array(tempout)
    return tempout


def get_p_mjd(mjd):
    path_env_data = 'env_data/'
    p_vt_s1 = pd.read_csv(path_env_data+'VIS-VT-S1.dat', sep=',')
    p_date = p_vt_s1.loc[p_vt_s1['mjd'] < float(mjd) + 0.05]
    p_date = p_date.loc[p_date['mjd'] > float(mjd) - 0.05]
    pdate = np.mean(p_date[' p'].values)

    return pdate