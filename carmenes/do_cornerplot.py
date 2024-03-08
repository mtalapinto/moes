import warnings
import glob
import chromatic_aberrations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.time import Time

import joint_aberrations
import optical_aberrations
import poly_fit
warnings.filterwarnings("ignore")
from pylab import *
from optics import vis_spectrometer
from simanneal import Annealer
import numpy as np
from optics import parameters
from optics import env_data
import pandas as pd
import ws_load
import aberration_corrections
from optics import echelle_orders
import os
import utils
import dynesty
import ws_load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TkAgg')
import corner


def do_plot(date):
    print('Doing cornerplot',)
    samples_names = [r'Ech. G [mm$^{-1}$]', r'Coll x-tilt [deg]', r'Ech. inc. angle [deg]',
                     r'Grism x-tilt [deg]', r'Cam x-tilt [deg]', r'Coll y-tilt [deg]',
                     r'Ech. $\gamma$-angle [deg]', r'Slit x-dec [mm]', r'Field flattener y-dec [mm]']

    samples = pd.read_csv('data/posteriors/'+str(date)+'/mc_samples.tsv', sep=',')
    best_ns_pars = pd.read_csv('data/posteriors/'+str(date)+'/best_fit_params_a.tsv', names=['pars'])
    #samples_plot = pd.DataFrame()
    #samples_plot[r'CCD T$_z$(deg)'] = samples[r'CCD T$_z$(deg)']
    #samples_plot[r'G$_{ech}$(mm$^{-1}$)'] = samples[r'G$_{ech}$(mm$^{-1}$)']
    #samples_plot[r'$\theta_{Blaze}$(m/s)'] = samples[r'$\theta_{Blaze}$(m/s)']
    #samples_plot[r'Coll T$_x$(deg)'] = samples[r'Coll T$_x$(deg)']
    #samples_plot[r'$\gamma_{ech}$(deg)'] = samples[r'$\gamma_{ech}$(deg)']
    #samples_plot[r'Coll T$_y$ (deg)'] = samples[r'Coll T$_y$ (deg)']
    #samples_plot[r'CCD-FF T$_z$ (deg)'] = samples[r'CCD-FF T$_z$ (deg)']
    #samples_plot[r'TM T$_{y}$ (deg)'] = samples[r'TM T$_{y}$ (deg)']
    #samples_plot[r'Cam. T$_{x}$(deg)'] = samples[r'Cam. T$_{x}$(deg)']
    samples_names_plot = samples_names.copy()
    pars_index = [11, 9, 12, 21, 28, 10, 13, 1, 32]
    pars = parameters.load_date('A', date)
    best_fit_sa = []
    for k in range(len(pars_index)):
        best_fit_sa.append(pars[pars_index[k]])

    # best_fit_par_dyn = [-2.08231, 13.37869, 40.68816416, 456.53436023, 0.03549355, 70.53001648, 153.82247384, 13.90580052, 2289.06760189, 0.27982054, 253.14207864, 11.37663006]
    # best_fit_par_stab = [-2.24, 13.22, 40.09, 456.81, 0.05, 89.68, 134.72, 12.59, 2369.76, 0.22, 210.58, 62.22]
    quantiles = corner.quantile(samples, 0.6827)
    #quantiles = corner.quantile(samples, -0.6827)
    print(best_ns_pars, best_fit_sa)
    plt.clf()
    figure = corner.corner(samples,
                           labels=samples_names_plot,
                           bins=25,
                           color='k',
                           reverse=False,
                           levels=(0.6827, 0.9545, 0.9973),
                           plot_contours=True,
                           quantiles=quantiles,
                           show_titles=True,
                           title_fmt=".2e",
                           title_kwargs={"fontsize": 13},
                           truths=best_fit_sa,
                           dpi=400,
                           truth_color='r',
                           scale_hist=True,
                           no_fill_contours=True,
                           plot_datapoints=True,
                           use_math_text=True,
                           label_kwargs={"position": (0, -1.), "labelpad": 0., "fontsize": 14},
                           max_n_ticks=3,
                           )

    fig2 = corner.corner(samples,
                         fig=figure,
                         labels=samples_names_plot,
                         bins=25,
                         color='k',
                         reverse=False,
                         levels=(0.6827, 0.9545, 0.9973),
                         plot_contours=False,
                         show_titles=False,
                         quantiles=quantiles,
                         # title_fmt=".2e",
                         # title_kwargs={"fontsize": 13},
                         truths=best_ns_pars['pars'].values,
                         dpi=400,
                         truth_color='b',
                         scale_hist=False,
                         no_fill_contours=True,
                         plot_datapoints=True,
                         use_math_text=False,
                         # label_kwargs={"position": (0, -1.), "labelpad": 0., "fontsize": 14},
                         # max_n_ticks=3,
                         )
    i = 0
    scalar_index = []
    fixed_index = []
    # Here we adjust the position of the labels, still need to format the ticks
    for ax in figure.get_axes():
        if not bool(ax.get_title()):
            xlabel = ax.xaxis.get_label()
            ylabel = ax.yaxis.get_label()
            xlabelbool = bool(xlabel.get_text())
            ylabelbool = bool(ylabel.get_text())
            if xlabelbool:
                formattick = str(ax.xaxis.get_major_formatter())[19:-26]
                xlabel.set_position((0.5, -0.4))
                # ax.ticklabel_format(style='sci')
                xlabelstr = xlabel.get_text()[:5]
                if xlabelstr == 'CCD T':
                    ylabel.set_position((-0.4, 0.5))

            elif ylabelbool:
                ylabel.set_position((-0.3, 0.5))

        else:
            titleax = ax.get_title()[:3]
            if titleax == 'Cam':
                xlabel = ax.xaxis.get_label()
                xlabel.set_position((0.5, -0.4))
                ylabel = ax.yaxis.get_label()
        i += 1

    figure.subplots_adjust(hspace=0.2)
    plt.savefig('data/posteriors/'+str(date)+'/corner_final.png', bbox_inches='tight')
    plt.show()
    print('... done.')


def error_budget_list(date):
    fib = 'A'
    spec = echelle_orders.init()  # ws_load.spectrum_from_data(data)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model_ref = vis_spectrometer.tracing(spec, init_state, fib, temps)
    pars = np.arange(0, 43, 1)
    parray, stdarr, rmsy = [], [], []
    for par in pars:
        params = parameters.load_date(fib, date)
        # params[-1] = pressure
        params_up = params.copy()
        params_lo = params.copy()

        # print(par)
        if par == 12:
            delta = 4.62e-4
        elif par == 30:
            delta = 9.62e-4
        else:
            delta = 6.09e-4

        delta = 4.2e-4
        params_up[par] += delta
        params_lo[par] -= delta
        # print('Parameter = ', parameters.get_name(par))
        # print(params[par], params_up[par], params_lo[par])
        parray.append(parameters.get_name(par, date))

        ws_mod_up = vis_spectrometer.tracing(spec, params_up, 'A', temps)
        ws_mod_lo = vis_spectrometer.tracing(spec, params_lo, 'A', temps)
        delta_models_up = ws_mod_up['x'] - model_ref['x']
        delta_models_lo = ws_mod_lo['x'] - model_ref['x']
        delta_models_up_y = ws_mod_up['y'] - model_ref['y']
        delta_models_lo_y = ws_mod_lo['y'] - model_ref['y']

        dec_x_up = np.mean(delta_models_up)
        dec_y_up = np.mean(delta_models_up_y)
        dec_x_lo = np.mean(delta_models_lo)
        dec_y_lo = np.mean(delta_models_lo_y)

        rms_up = np.sqrt(np.sum(delta_models_up ** 2) / len(delta_models_up))
        rms_lo = np.sqrt(np.sum(delta_models_lo ** 2) / len(delta_models_lo))
        rms_up_y = np.sqrt(np.sum(delta_models_up_y ** 2) / len(delta_models_up_y))
        rms_lo_y = np.sqrt(np.sum(delta_models_lo_y ** 2) / len(delta_models_lo_y))
        std_up = np.std(delta_models_up)
        std_lo = np.std(delta_models_lo)
        std_up_y = np.std(delta_models_up_y)
        std_lo_y = np.std(delta_models_lo_y)
        std_x = (std_up + std_lo) / 2.
        std_y = (std_up_y + std_lo_y) / 2.
        std = (std_x + std_y) / 2.
        rms = (rms_up + rms_lo) / 2. + (dec_x_up + dec_x_lo) / 2
        rms_y = (rms_up_y + rms_lo_y) / 2.
        stdarr.append(rms)
        rmsy.append(rms_y)
        ws_mod_up['wave'] = ws_mod_up['wave'] * 1e4
        # plt.figure(figsize=[10, 4])
        # plt.plot(ws_mod_up['wave'], delta_models_up, 'r.')
        # plt.plot(ws_mod_up['wave'], delta_models_lo, 'b.')
        # plt.show()
        # plt.clf()
        # plt.close()

    out = pd.DataFrame()
    out['name'] = parray
    out['rms_x'] = stdarr
    out['rms_y'] = rmsy

    out = out.sort_values(by='rms_x')
    print(out)

    '''
        if par == 42:
            axes[0, 0].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[0, 0].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[0, 0].set_title('CCD z-tilt')
            axes[0, 0].set_ylim(-60., 60.)
        elif par == 11:
            axes[0, 1].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[0, 1].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[0, 1].set_title(r'Echelle $G$')
            axes[0, 1].set_ylim(-20., 20.)

        elif par == 12:
            axes[0, 2].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[0, 2].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[0, 2].set_title('Echelle blaze angle')
            axes[0, 2].set_ylim(-2., 2.)

        elif par == 9:
            axes[1, 0].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[1, 0].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[1, 0].set_title('Collimator x-tilt')
            axes[1, 0].set_ylim(-2.5, 2.5)

        elif par == 13:
            axes[1, 2].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[1, 2].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[1, 2].set_title(r'Echelle $\gamma$-angle')
            axes[1, 2].set_ylim(-0.25, 0.25)

        elif par == 10:
            axes[1, 1].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[1, 1].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[1, 1].set_title('Collimator y-tilt')
            axes[1, 1].set_ylim(-0.5, 0.5)

        elif par == 35:
            axes[2, 1].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[2, 1].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[2, 1].set_title('Field flattener z-tilt')
            axes[2, 1].set_xlabel(r'$\lambda$ [$\rm \AA{}$]', fontsize='x-large', family='serif')

        elif par == 17:
            axes[2, 2].scatter(ws_mod_up[:, 1], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[2, 2].scatter(ws_mod_up[:, 1], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[2, 2].set_title('Transfer mirror y-tilt')

        elif par == 28:
            axes[2, 0].scatter(ws_mod_up[:, 1], delta_models_up-0.2, color='blue', marker='.', s=2, alpha=0.7)
            axes[2, 0].scatter(ws_mod_up[:, 1], delta_models_lo+0.2, color='red', marker='.', s=2, alpha=0.7)
            axes[2, 0].set_title('Camera x-tilt')
            axes[2, 0].set_ylim(-0.1, 0.1)
    '''

    # plt.scatter(ws_mod_up[:,1], delta_models_up, color='blue')
    # plt.scatter(ws_mod_up[:,1], delta_models_lo, color='red')
    # plt.xlim(0, 4096)

    # plt.tight_layout(pad=0.5)
    # plt.savefig('plots/error_budget_vis.png')
    # plt.show()
    # plt.clf()

    # plt.plot(ws_mod[:,2], ws_mod[:,3],'r.')
    # plt.show()
    # print(params)
    # print(spectrum)
    # print(temps)


def error_budget_plots(date):
    fib = 'A'
    spec = echelle_orders.init()  # ws_load.spectrum_from_data(data)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model_ref = vis_spectrometer.tracing(spec, init_state, fib, temps)
    pars = [11, 9, 12, 21, 28, 10, 13, 1, 32]

    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(11, 8))
    fig.text(-0.001, 0.55, r'$\rm \delta$x$_\mathrm{T}$ (pix)', va='center', rotation='vertical', fontsize='x-large',
             family='serif')
    # plt.subplots_adjust(left=0.5,
    #                bottom=0.5,
    #                right=0.9,
    ##                top=0.9,
    #               wspace=0.001,
    #               hspace=0.001)
    parray = []
    for par in pars:
        params = parameters.load_date(fib, date)
        # params[-1] = pressure
        params_up = params.copy()
        params_lo = params.copy()

        # print(par)
        if par == 12:
            delta = 4.62e-4
        elif par == 30:
            delta = 9.62e-4
        else:
            delta = 6.09e-4

        delta = 4.2e-4
        params_up[par] += delta
        params_lo[par] -= delta
        # print('Parameter = ', parameters.get_name(par))
        # print(params[par], params_up[par], params_lo[par])
        parray.append(parameters.get_name(par, date))

        ws_mod_up = vis_spectrometer.tracing(spec, params_up, 'A', temps)
        ws_mod_lo = vis_spectrometer.tracing(spec, params_lo, 'A', temps)
        delta_models_up = ws_mod_up['x'] - model_ref['x']
        delta_models_lo = ws_mod_lo['x'] - model_ref['x']
        delta_models_up_y = ws_mod_up['y'] - model_ref['y']
        delta_models_lo_y = ws_mod_lo['y'] - model_ref['y']

        ws_mod_up['wave'] = ws_mod_up['wave'] * 1e4
        if par == 11:
            axes[0, 0].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[0, 0].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[0, 0].set_title('Echelle constant $G$')
            axes[0, 0].set_ylim(-10., 10.)
        elif par == 9:
            axes[0, 1].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[0, 1].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[0, 1].set_title(r'Collimator x-tilt')
            axes[0, 1].set_ylim(-2.5, 2.5)

        elif par == 12:
            axes[0, 2].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[0, 2].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[0, 2].set_title('Echelle incidence angle')
            axes[0, 2].set_ylim(-1., 1.)

        elif par == 21:
            mean_hi = np.median(delta_models_up)
            mean_lo = np.median(delta_models_lo)
            label_hi = r'$\overline{\delta x}_{+0.3^{\circ}C}$ = ' + str(np.round(mean_hi, 2)) + ' pix'
            label_lo = r'$\overline{\delta x}_{-0.3^{\circ}C}$ = ' + str(np.round(mean_lo, 2)) + ' pix'
            axes[1, 0].scatter(ws_mod_up['wave'], delta_models_up - mean_hi, color='blue', marker='.', s=2, alpha=0.7,
                               label=label_hi)
            axes[1, 0].scatter(ws_mod_up['wave'], delta_models_lo - mean_lo, color='red', marker='.', s=2, alpha=0.7,
                               label=label_lo)
            axes[1, 0].set_title('Grism x-tilt')
            # axes[1, 0].set_ylabel(r'$\rm \delta$x$_T$ [pix]')
            axes[1, 0].set_ylim(-0.012, 0.012)
            axes[1, 0].legend(loc='best')


        elif par == 28:
            mean_hi = np.median(delta_models_up)
            mean_lo = np.median(delta_models_lo)
            label_hi = r'$\overline{\delta x}_{+0.3^{\circ}C}$ = ' + str(np.round(mean_hi, 2)) + ' pix'
            label_lo = r'$\overline{\delta x}_{-0.3^{\circ}C}$ = ' + str(np.round(mean_lo, 2)) + ' pix'

            axes[1, 1].scatter(ws_mod_up['wave'], delta_models_up - mean_hi,
                               color='blue', marker='.', s=2, alpha=0.7,
                               label=label_hi)
            # axes[1, 1].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.',
            #                   s=2, alpha=0.7,
            #                   label=r'$\bar{\delta x_{T^+}}$ = ' + str(np.round(np.mean(delta_models_up), 2)) + 'pix')
            axes[1, 1].scatter(ws_mod_up['wave'], delta_models_lo - mean_lo, color='red', marker='.',
                               s=2, alpha=0.7, label=label_lo)
            # axes[1, 1].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.',
            #                   s=2, alpha=0.7,
            #                   label=r'$\bar{\delta x_{T^-}}$ = ' + str(np.round(np.mean(delta_models_lo), 2)) + 'pix')
            axes[1, 1].set_title(r'Camera x-tilt')
            axes[1, 1].set_ylim(-0.012, 0.012)
            # axes[1, 1].set_ylim(-0.24, 0.24)
            axes[1, 1].legend(loc='best')


        elif par == 10:
            axes[1, 2].scatter(ws_mod_up['wave'], delta_models_up - np.mean(delta_models_up), color='blue', marker='.',
                               s=2, alpha=0.7,
                               label=r'$\bar{\delta x_{T^+}}$ = ' + str(np.round(np.mean(delta_models_up), 2)))
            # axes[1, 2].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.',
            #                   s=2, alpha=0.7,
            #                   label=r'$\bar{\delta x_{T^+}}$ = ' + str(np.round(np.mean(delta_models_up), 2)))
            axes[1, 2].scatter(ws_mod_up['wave'], delta_models_lo - np.mean(delta_models_lo), color='red', marker='.',
                               s=2, alpha=0.7,
                               label=r'$\bar{\delta x_{T^-}}$ = ' + str(np.round(np.mean(delta_models_lo), 2)))
            # axes[1, 2].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.',
            #                   s=2, alpha=0.7,
            #                   label=r'$\bar{\delta x_{T^-}}$ = ' + str(np.round(np.mean(delta_models_lo), 2)))
            axes[1, 2].set_title(r'Collimator y-tilt')
            axes[1, 2].set_ylim(-0.25, 0.25)
            # axes[1, 2].set_ylim(-0.24, 0.24)
            # axes[1, 2].legend(markerscale=5, fontsize=12)


        elif par == 13:
            axes[2, 0].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[2, 0].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[2, 0].set_title(r'Echelle $\gamma$-angle')

        elif par == 1:
            axes[2, 1].scatter(ws_mod_up['wave'], delta_models_up, color='blue', marker='.', s=2, alpha=0.7)
            axes[2, 1].scatter(ws_mod_up['wave'], delta_models_lo, color='red', marker='.', s=2, alpha=0.7)
            axes[2, 1].set_title('Slit x-decenter')
            axes[2, 1].set_ylim(-0.12, 0.12)
            axes[2, 1].set_xlabel(r'$\lambda$ [$\rm \AA{}$]', fontsize='x-large', family='serif')


        elif par == 32:
            mean_hi = np.median(delta_models_up)
            mean_lo = np.median(delta_models_lo)
            label_hi = r'$\overline{\delta x}_{+0.3^{\circ}C}$ = ' + str(np.round(mean_hi, 2)) + ' pix'
            label_lo = r'$\overline{\delta x}_{-0.3^{\circ}C}$ = ' + str(np.round(mean_lo, 2)) + ' pix'
            axes[2, 2].scatter(ws_mod_up['wave'], delta_models_up - mean_hi, color='blue', marker='.', s=2, alpha=0.7,
                               label=label_hi)
            axes[2, 2].scatter(ws_mod_up['wave'], delta_models_lo - mean_lo, color='red', marker='.', s=2, alpha=0.7,
                               label=label_lo)
            axes[2, 2].set_title('Field flattener y-decenter')
            axes[2, 2].legend(loc='best')
        # plt.scatter(ws_mod_up[:,1], delta_models_up, color='blue')
        # plt.scatter(ws_mod_up[:,1], delta_models_lo, color='red')
        # plt.xlim(0, 4096)
    plt.tight_layout(pad=0.5)
    plt.savefig('plots/error_budget_vis.png')
    plt.show()
    plt.clf()

    # plt.plot(ws_mod[:,2], ws_mod[:,3],'r.')
    # plt.show()
    # print(params)
    # print(spectrum)
    # print(temps)


if __name__ == '__main__':
    date = '2018-11-22'
    #do_plot(date)
    error_budget_list(date)


'''
fig = corner.corner(no_stab_samples,
bins=25, 
color="r", 
reverse=False, 
#upper= True, 
#labels=samples_names, 
#quantiles=[0.1585, 0.8415],
levels=(0.6827, 0.9545,0.9973),
#smooth=1.0, 
#smooth1d=1.0,
plot_contours=False, 
#show_titles=True, 
truths=best_fit_par_stab, 
dpi = 200, 
#pad=15, 
#labelpad = 50 ,
truth_color ='g', 
#title_kwargs={"fontsize": 12}, 
scale_hist=True,  
no_fill_contours=True, 
plot_datapoints=False)

corner.corner(stab_samples,
fig = fig,
bins=25, 
color="k", 
reverse=False, 
upper= True, 
#labels=samples_names, 
quantiles=[0.1585, 0.8415],
levels=(0.6827, 0.9545,0.9973),
smooth=1.0, 
smooth1d=1.0,
plot_contours= True, 
show_titles=False, 
truths=best_fit_par_dyn, 
dpi = 200, 
pad=25, 
labelpad = 50 ,
label_kwargs={"fontsize": 26}, 
truth_color ='blue',
title_kwargs={"fontsize": 12}, 
scale_hist=True,  
no_fill_contours=True, 
plot_datapoints=False)
plt.xticks(fontsize=14)
    
#plt.show()
fig.savefig("hd25723_stab_cornerplot.pdf")

#print(samples)
#samples = samples.drop(r'Grism T$_{x}$(deg)', inplace=True, axis=1)
#print(samples)
#samples = samples.drop(r'CCD-FF T$_{x}$(deg)', inplace=True, axis=1)
#samples = samples.drop(r'Grism Apex~[(deg)', inplace=True, axis=1)
#samples = samples.drop(r'Cam. T$_{y}$', inplace=True, axis=1)
#samples = samples.drop(r'Grism Apex~[(deg)', inplace=True, axis=1)
#samples = samples.drop(r'd$_{FF-CCD}$~(mm)', inplace=True, axis=1)
#samples = samples.drop(r'CCD defocus~(mm)', inplace=True, axis=1)


'''

'''
formattick = str(ax.xaxis.get_major_formatter())[19:-26]
# print(formattick)
if formattick == 'ScalarFormatter':
    scalar_index.append(i)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True))


elif formattick == 'FixedFormatter':
    fixed_index.append(i)
else:
    print('mierdofilo')
'''

