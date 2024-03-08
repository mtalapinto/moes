import warnings
import glob
import chromatic_aberrations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
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
from scipy.stats import spearmanr
import ws_load
import aberration_corrections
import os
import utils
import dynesty
import ws_load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import optimization
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
matplotlib.rc('font', **font)


def get_spec_type(starid):
    carmencita = pd.read_csv('data/carmencita.097.csv', sep=',')
    stardata = carmencita.loc[carmencita['Karmn'] == starid]
    return stardata['SpT'].values


def get_sum(vec):
    fvec = np.sort(vec)
    fval = np.median(fvec)
    nn = int(np.around(len(fvec) * 0.15865))
    vali, valf = fval - fvec[nn], fvec[-nn] - fval
    return fval, vali, valf


def get_rvscale(spectrum, params_a, params_b, temps, date):
    nwaves = 2
    dwave = 5.e-6
    # jd = mjd + 2400000.5
    spectrum_aux = np.copy(spectrum)
    spectrum_aux[:, 1] = spectrum_aux[:, 1] + dwave

    # Drifted models
    wsa_model = vis_spectrometer.tracing(spectrum_aux, params_a, 'A', temps)

    chromatic_coeffs_a = chromatic_aberrations.load_coeffs(date, 'a')
    chromatic_model_a = chromatic_aberrations.function(np.array(wsa_model['wave'].values), chromatic_coeffs_a)
    wsa_model['x'] = wsa_model['x'] + chromatic_model_a
    optical_coeffs_a = optical_aberrations.load_coeffs(date, 'a')
    optical_model_a = optical_aberrations.function_seidel(wsa_model, optical_coeffs_a)
    wsa_model['x'] = wsa_model['x'] + optical_model_a

    polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date + '/'

    poly_coefs_a = pd.read_csv(polydir + 'poly_coefs_a.csv', sep=',')
    z = [poly_coefs_a['p0'].values[0],
         poly_coefs_a['p1'].values[0],
         poly_coefs_a['p2'].values[0],
         poly_coefs_a['p3'].values[0],
         poly_coefs_a['p4'].values[0]]
    f = np.poly1d(z)
    polymodela = f(wsa_model['x'].values)
    wsa_model['x'] = wsa_model['x'] + polymodela

    wsam = pd.DataFrame()
    wsam['order'] = wsa_model['order'].values
    wsam['wave'] = wsa_model['wave'].values
    wsam['x'] = wsa_model['x'].values

    wsb_model = vis_spectrometer.tracing(spectrum_aux, params_b, 'B', temps)
    chromatic_coeffs_b = chromatic_aberrations.load_coeffs(date, 'b')
    chromatic_model_b = chromatic_aberrations.function(np.array(wsb_model['wave'].values), chromatic_coeffs_b)
    wsb_model['x'] = wsb_model['x'] + chromatic_model_b
    optical_coeffs_b = optical_aberrations.load_coeffs(date, 'b')
    optical_model_b = optical_aberrations.function_seidel(wsb_model, optical_coeffs_b)
    wsb_model['x'] = wsb_model['x'] + optical_model_b

    polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date + '/'

    poly_coefs_b = pd.read_csv(polydir + 'poly_coefs_b.csv', sep=',')
    z = [poly_coefs_b['p0'].values[0],
         poly_coefs_b['p1'].values[0],
         poly_coefs_b['p2'].values[0],
         poly_coefs_b['p3'].values[0],
         poly_coefs_b['p4'].values[0]]
    f = np.poly1d(z)
    polymodelb = f(wsb_model['x'].values)
    wsb_model['x'] = wsb_model['x'] + polymodelb

    wsbm = pd.DataFrame()
    wsbm['order'] = wsb_model['order'].values
    wsbm['wave'] = wsb_model['wave'].values
    wsbm['x'] = wsb_model['x'].values

    # Non drifted models
    wsa_model_ref = vis_spectrometer.tracing(spectrum, params_a, 'A', temps)

    chromatic_coeffs_a = chromatic_aberrations.load_coeffs(date, 'a')
    chromatic_model_a = chromatic_aberrations.function(np.array(wsa_model_ref['wave'].values), chromatic_coeffs_a)
    wsa_model_ref['x'] = wsa_model_ref['x'] + chromatic_model_a
    optical_coeffs_a = optical_aberrations.load_coeffs(date, 'a')
    optical_model_a = optical_aberrations.function_seidel(wsa_model_ref, optical_coeffs_a)
    wsa_model_ref['x'] = wsa_model_ref['x'] + optical_model_a

    poly_coefs_a = pd.read_csv(polydir + 'poly_coefs_a.csv', sep=',')
    z = [poly_coefs_a['p0'].values[0],
         poly_coefs_a['p1'].values[0],
         poly_coefs_a['p2'].values[0],
         poly_coefs_a['p3'].values[0],
         poly_coefs_a['p4'].values[0]]
    f = np.poly1d(z)
    polymodela = f(wsa_model_ref['x'].values)
    wsa_model_ref['x'] = wsa_model_ref['x'] + polymodela

    wsa0 = pd.DataFrame()
    wsa0['order'] = wsa_model_ref['order'].values
    wsa0['wave'] = wsa_model_ref['wave'].values
    wsa0['x'] = wsa_model_ref['x'].values

    wsb_model_ref = vis_spectrometer.tracing(spectrum, params_b, 'B', temps)

    chromatic_coeffs_b = chromatic_aberrations.load_coeffs(date, 'b')
    chromatic_model_b = chromatic_aberrations.function(np.array(wsb_model_ref['wave'].values), chromatic_coeffs_b)
    wsb_model_ref['x'] = wsb_model_ref['x'] + chromatic_model_b
    optical_coeffs_b = optical_aberrations.load_coeffs(date, 'b')
    optical_model_b = optical_aberrations.function_seidel(wsb_model_ref, optical_coeffs_b)
    wsb_model_ref['x'] = wsb_model_ref['x'] + optical_model_b

    poly_coefs_b = pd.read_csv(polydir + 'poly_coefs_b.csv', sep=',')
    z = [poly_coefs_b['p0'].values[0],
         poly_coefs_b['p1'].values[0],
         poly_coefs_b['p2'].values[0],
         poly_coefs_b['p3'].values[0],
         poly_coefs_b['p4'].values[0]]
    f = np.poly1d(z)
    polymodelb = f(wsb_model_ref['x'].values)
    wsb_model_ref['x'] = wsb_model_ref['x'] + polymodelb

    wsb0 = pd.DataFrame()
    wsb0['order'] = wsb_model_ref['order'].values
    wsb0['wave'] = wsb_model_ref['wave'].values
    wsb0['x'] = wsb_model_ref['x'].values

    dpix_a = np.abs(wsam['x'].values - wsa0['x'].values)
    dpix_b = np.abs(wsbm['x'].values - wsb0['x'].values)
    #print(dpix_a, dpix_b)
    #dpix_a_mean = np.mean(dpix_a)
    #dpix_b_mean = np.mean(dpix_b)

    dwavec = dwave * 3.e8
    drv_a = np.abs(dwavec / spectrum[:, 1] / dpix_a)
    drv_b = np.abs(dwavec / spectrum[:, 1] / dpix_b)

    return drv_a, drv_b


def nzp(bjd):
    nzp_data = pd.read_csv('gtoc_vis_night_zero.txt', sep=' ',
                           names=['jd', 'nzp', 'sigma_nzp', 'nstars', 'flag'])
    bjd_int = int(bjd)
    nzp_bjd = nzp_data.loc[nzp_data['jd'] == bjd_int]
    if len(nzp_bjd) == 0:
        nzp_out = 0.
        nzp_out_sigma = 0.
    else:
        nzp_out = float(nzp_bjd['nzp'].values[0])
        nzp_out_sigma = float(nzp_bjd['sigma_nzp'].values[0])

    return nzp_out, nzp_out_sigma


def gaus(x, height, x0, sigma):
    return height * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))  # + offset


def sine(x, amp, phase, P, offset):
    return amp * np.sin(2*np.pi/P * x + phase) + offset


def residuals_plots(date, fib):
    # Load data
    kin = 'hcl'
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    data = ws_load.read_ws(date, kin, fib)
    spec = ws_load.spectrum_from_data(data)

    #init_state = parameters.load_date(fib, date)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model = vis_spectrometer.tracing(spec, init_state, fib, temps)
    residuals_instrument = data['posm'].values - model['x'].values

    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model
    residuals_chromatic = data['posm'].values - model['x'].values

    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model
    model = optimization.poly_fit_date(data, model, 4)
    residuals_optical = data['posm'].values - model['x'].values
    X = data['posm'].values
    wave = model['wave'].values

    outdir = 'plots/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #plt.figure(figsize=[8, 3])
    rms_caracal = np.sqrt(np.sum((data['posm'].values - data['posc'].values) ** 2) / len(data['posm'].values))
    rms_moes = np.sqrt(np.sum((data['posm'].values - model['x'].values) ** 2) / len(data['posm'].values))

    c = model['wave'].values  # ((ws_model[:,1] - wave_min))/dwave

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0], xticklabels=[], xticks=[], axisbelow=True)
    ax1 = fig.add_subplot(gs[1, 0])

    print('rms caracal = ', rms_caracal, 'rms moes = ', rms_moes, ', fiber = ', fib)
    ax0.plot(data['posm'].values, data['posm'].values - data['posc'].values, 'b.', alpha=0.5, label='CARACAL')
    ax0.plot(data['posm'].values, data['posm'].values - model['x'].values, 'r.', alpha=0.5, label=r'moes')
    ax0.legend()
    ax0.set_xlim(0, 4096)
    ax0.set_xlabel(r'$x_{m}$ (pix)')
    ax0.set_ylabel(r'$x_{moes}$ - $x_{m}$ (pix)')
    # plt.scatter(ws_model[:, 2], ws_model[:, 3], marker='.', c=c, cmap='inferno', s=8)
    ax0.set_ylim(-0.2, 0.2)
    ax1.plot(model['wave'].values, data['posm'].values - data['posc'].values, 'b.', alpha=0.5, label='CARACAL')
    ax1.plot(model['wave'].values, data['posm'].values - model['x'].values, 'r.', alpha=0.5, label=r'moes')
    ax1.legend()
    ax1.set_xlim(min(model['wave'].values), max(model['wave'].values))
    ax1.set_xlabel(r'$\lambda$ [$\rm \AA{}$]')
    ax1.set_ylabel(r'$x_{moes}$ - $x_{m}$ [pix]')
    ax1.set_ylim(-0.2, 0.2)
    plt.show()
    plt.savefig(outdir + 'residuals_' + str(date) + '_' + str(fib) + '.png', bbox_tight=True)
    plt.clf()
    plt.close()
    return rms_caracal, rms_moes


def full_model_residuals(date, fib):
    # Load data
    kin = 'hcl'
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    data = ws_load.read_ws(date, kin, fib)
    spec = ws_load.spectrum_from_data(data)

    # init_state = parameters.load_date(fib, date)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model = vis_spectrometer.tracing(spec, init_state, fib, temps)
    residuals_instrument = data['posm'].values - model['x'].values
    rms_I = np.sqrt(np.sum(residuals_instrument ** 2) / len(residuals_instrument))

    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model
    residuals_chromatic = data['posm'].values - model['x'].values
    rms_IC = np.sqrt(np.sum(residuals_chromatic ** 2) / len(residuals_instrument))

    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model
    print(model)
    print(data)
    model, rms_ICO = optimization.poly_fit_date(data, model, 4)
    residuals_optical = data['posm'].values - model['x'].values
    X = data['posm'].values
    wave = model['wave'].values

    marksize = 14
    c = wave*1e4
    # plt.figure(figsize=(8,20))
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=True, figsize=(12, 16),
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    axes[0].scatter(X, residuals_instrument, c=c, marker='o', s=marksize, alpha=0.8,
                    label='Instrument post-fit residuals, rms = '+str(np.round(rms_I, 3))+' pix', cmap='inferno')
    axes[0].set_xlim(0, 4096)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].legend(markerscale=2)
    axes[0].set_ylabel(r'$x_k$ - $x_{RT}$ [pix]', fontsize='large', family='serif')
    axes[0].set_xlabel(r'$x_k$ [pix]', fontsize='large', family='serif')


    axes[1].scatter(wave*1e4, residuals_instrument, c=c, marker='o', s=marksize, cmap = 'inferno', alpha=0.8)
    axes[1].scatter(wave*1e4, chromatic_model, color='blue', marker='o', s=marksize, label='Chromatic aberration model')
    axes[1].set_xlim(min(wave)*1e4, max(wave)*1e4)
    axes[1].legend(loc=4, markerscale=2)
    axes[1].set_ylabel(r'$x_k$ - $x_\mathrm{RT}$ [pix]', fontsize='large', family='serif')
    axes[1].set_xlabel(r'$\lambda$ [$\rm \AA{}$]', fontsize='large', family='serif')
    axes[1].set_ylim(-0.5, 0.5)

    axes[2].scatter(X, residuals_chromatic, c=c, marker='o', s=marksize, cmap='inferno', alpha=0.8,
                    label='Instrument+chromatic post-fit residuals, rms = '+str(np.round(rms_IC, 3))+' pix')
    axes[2].scatter(X, optical_model, color='blue', marker='o', s=marksize, alpha=0.2, label='Seidel aberration model')
    axes[2].set_xlim(0, 4096)
    axes[2].legend(markerscale=2)
    axes[2].set_ylabel(r'$x_k$ - $x_\mathrm{C}$ [pix]', fontsize='large', family='serif')
    axes[2].set_xlabel(r'$x_k$ [pix]', fontsize='large', family='serif')
    axes[2].set_ylim(-0.25, 0.3)

    axes[3].scatter(X, residuals_optical, c=c, marker='o', s=marksize, alpha=0.8, cmap='inferno',
                    label='Instrument+chromatic+Seidel aberration post-fit residuals, rms = '+str(np.round(rms_ICO, 3))+' pix', zorder=20)
    #axes[3].scatter(X, data['posm'].values - data['posc'].values, c='blue', marker='o', s=marksize, alpha=0.2, cmap='inferno',
    #                label='CARACAL post-fit residuals, rms = ' + str(
    #                    np.round(np.sqrt(np.sum(data['posm'].values - data['posc'].values) / len(data['posm'].values)), 3)) + ' pix', zorder=0)
    axes[3].set_xlim(0, 4096)
    # axes[2].set_legend('Instrument + chromatic + optical aberrations fit residuals')
    axes[3].set_xlabel(r'$x_{k}$ [pix]', fontsize='large', family='serif')
    axes[3].set_ylabel(r'$x_k$ - $x_{\rm moes}$ [pix]', fontsize='large', family='serif')
    axes[3].legend(markerscale=2)
    axes[3].set_ylim(-0.25, 0.3)

    plt.tight_layout()
    #plt.savefig('plots/post_fit_residuals_vis.png')
    plt.show()
    plt.clf()
    plt.close()


def echellogram_plot(date, fib):
    # Load data
    kin = 'hcl'
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    data = ws_load.read_ws(date, kin, fib)
    spec = ws_load.spectrum_from_data(data)

    # init_state = parameters.load_date(fib, date)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model = vis_spectrometer.tracing(spec, init_state, fib, temps)

    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model

    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model
    model, rms = optimization.poly_fit_date(data, model, 4)
    residuals_optical = data['posm'].values - model['x'].values
    residuals_data = data['posm'].values - data['posc'].values
    X = data['posm'].values
    wave = model['wave'].values

    outdir = 'plots/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # plt.figure(figsize=[8, 3])
    rms_caracal = np.sqrt(np.sum((data['posm'].values - data['posc'].values) ** 2) / len(data['posm'].values))
    rms_moes = np.sqrt(np.sum((data['posm'].values - model['x'].values) ** 2) / len(data['posm'].values))

    c = model['wave'].values  # ((ws_model[:,1] - wave_min))/dwave

    fig = plt.figure(figsize=[10, 8])
    plt.scatter(model['x'].values, model['y'].values, marker='o', c=c, cmap='inferno', s=8)
    # plt.set_aspect('equal')
    plt.xlabel(r'$x$ [pix]', fontsize='large', family='serif')
    plt.ylabel(r'$y$ [pix]', fontsize='large', family='serif')
    plt.xlim(0, 4096)
    plt.ylim(0, 4096)
    m = cm.ScalarMappable(cmap=cm.inferno)
    m.set_array(model['wave'] * 1e4)
    # cax = fig.add_axes([0.96, .05, 0.1, 0.99])
    clb = fig.colorbar(m, orientation='vertical', fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\lambda$ [$\rm\AA{}$]', fontsize='large', family='serif')
    plt.tight_layout()

    #plt.savefig('plots/echellogram_vis.png')
    plt.show()
    plt.clf()
    plt.close()

    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=[14, 7])

    msize = 6
    axs[0, 0].plot(X, residuals_data, 'k.', alpha=0.4,
                   label=r'$x_c$ from $\tt caracal$, rms = ' + str(np.round(rms_caracal, 3)), markersize=msize,
                   zorder=0)
    axs[0, 0].plot(X, residuals_optical, 'r.', alpha=0.4,
                   label=r'$x_c$ from $\tt moes$, rms = ' + str(np.round(rms_moes, 3)), markersize=msize, zorder=1)
    axs[0, 0].set_xlim(0, 4096)
    axs[0, 0].set_xlabel(r'$x_k$ [pix]', fontsize='large', family='serif')
    axs[0, 0].set_ylabel(r'$x_k - x_c$ [pix]', fontsize='large', family='serif')
    axs[0, 0].legend(loc='best', prop={'size': 12}, markerscale=2.)
    # axs[0, 0].set_ylim()

    axs[1, 0].plot(wave, residuals_data, 'k.', alpha=0.4,
                   label=r'$x_{\rm caracal}$, rms = ' + str(np.round(rms_caracal, 3)), markersize=msize,
                   zorder=0)
    axs[1, 0].plot(wave, residuals_optical, 'r.', alpha=0.4,
                   label=r'$x_{\rm moes}$, rms = ' + str(np.round(rms_moes, 3)), markersize=msize, zorder=1)
    axs[1, 0].set_xlim(min(wave), max(wave))
    axs[1, 0].set_xlabel(r'$\lambda$ [$\rm\AA{}$]', fontsize='large', family='serif')
    axs[1, 0].set_ylabel(r'$x_k - x_{\rm model}$ [pix]', fontsize='large', family='serif')

    axs[0, 1].hist(residuals_optical, color='r', alpha=0.6, orientation='horizontal')
    # axs[1, 1].hist(wsa_data[:, 3] - wsa_data[:, 2], color='k', alpha=0.6, orientation='horizontal')
    # axs[1, 0].legend(loc='best', prop={'size': 12}, markerscale=2.)
    # axs[1, 0].set_ylim()

    plt.subplots_adjust(hspace=0.25)
    #plt.show()
    plt.clf()
    plt.close()
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    # ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
    fig, axs = plt.subplot_mosaic([['pix', 'hist']],
                                   #['wave', 'hist']],
                                  figsize=(14, 6), constrained_layout=True, gridspec_kw={'width_ratios': [4.2, 1]})

    axs['pix'].plot(X, residuals_data, 'b.', alpha=0.4,
                    label=r'$x_c$ from $\tt caracal$, rms = ' + str(np.round(rms_caracal, 3)), markersize=msize,
                    zorder=0)
    axs['pix'].plot(X, residuals_optical, 'r.', alpha=0.4,
                    label=r'$x_c$ from $\tt moes$, rms = ' + str(np.round(rms_moes, 3)), markersize=msize, zorder=1)
    axs['pix'].set_xlim(0, 4096)
    axs['pix'].set_ylim(-0.2, 0.2)
    axs['pix'].set_xlabel(r'$x_k$ [pix]', fontsize='large', family='serif')
    axs['pix'].set_ylabel(r'$x_k - x_c$ [pix]', fontsize='large', family='serif')
    axs['pix'].legend(loc=3, prop={'size': 12}, markerscale=2.)

    #axs['wave'].plot(wave * 1e4, residuals_data, 'b.', alpha=0.4,
    #                 label=r'$x_c$ from $\tt caracal$', markersize=msize * 1.1, zorder=0)
    #axs['wave'].plot(wave * 1e4, residuals_optical, 'r.', alpha=0.4, label=r'$x_c$ from ${\tt moes}$',
    #                 markersize=msize, zorder=1)
    #axs['wave'].set_xlim(min(wave * 1e4), max(wave * 1e4))
    #axs['wave'].set_xlabel(r'$\lambda$ [$\rm\AA{}$]')
    #axs['wave'].set_ylabel('$x_k - x_c$ [pix]', fontsize='large', family='serif')
    #axs['wave'].legend(loc='best', prop={'size': 14}, markerscale=2.)

    binwidth = (max(residuals_optical) - min(residuals_optical)) / 25.
    bins_all = np.arange(min(residuals_optical), max(residuals_optical), binwidth)

    nc, binsc, patchc = axs['hist'].hist(residuals_data, color='b', alpha=0.4,
                                         orientation='horizontal', bins=bins_all,
                                         edgecolor='black', label=r'$x_c$ from $\tt caracal$')
    nm, binsm, patchm = axs['hist'].hist(residuals_optical, color='r', alpha=0.4,
                                         orientation='horizontal', bins=bins_all, edgecolor='black',
                                         label=r'$x_c$ from $\tt moes$')
    yrange = np.arange(-0.15, 0.5, 0.001)
    poptc, pcovc = curve_fit(gaus, (binsc[1:] - (binsc[1] - binsc[0]) / 2), nc,
                             p0=[max(nc), np.mean(residuals_data),
                                 np.std(residuals_data)])
    poptm, pcovm = curve_fit(gaus, (binsm[1:] - (binsm[1] - binsm[0]) / 2), nm,
                             p0=[max(nm), np.mean(residuals_optical),
                                 np.std(residuals_optical)])
    fwhmm = 2 * np.sqrt(2 * np.log(2)) * poptm[2]
    fwhmc = 2 * np.sqrt(2 * np.log(2)) * poptc[2]
    print(poptm[2], poptc[2])
    # print(fwhmm, fwhmc)
    axs['hist'].plot(gaus(yrange, *poptc), yrange, 'b-', label=r'$\sigma_{CARACAL}~=~$' + str(0.031))
    axs['hist'].plot(gaus(yrange, *poptm), yrange, 'r-', label=r'$\sigma_{moes}~=~$' + str(np.round(poptm[2], 3)))
    axs['hist'].set_xlabel('Number of lines')
    axs['hist'].set_ylabel('$x_k - x_c$ [pix]')
    axs['hist'].yaxis.set_label_position("right")
    axs['hist'].legend(loc='best', prop={'size': 10}, markerscale=2.)
    axs['hist'].yaxis.tick_right()
    axs['hist'].set_ylim(-0.2, 0.2)
    #plt.savefig('plots/ws_compare_caracal_moes.png')
    plt.show()
    plt.clf()
    plt.close()


def make_dd_hcl_file():
    path_data = 'data/vis_ws_timeseries/'
    datesdirs = glob.glob(path_data + '*')
    dates, jd = [], []
    for date in datesdirs:
        d = date[len(path_data):]
        disot = Time(d + 'T12:00:00.0', format='isot')
        dates.append(d)
        jd.append(disot.jd)

    daux = pd.DataFrame()
    daux['dates'] = dates
    daux['jd'] = jd
    daux = daux.sort_values(by='jd')

    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])
    kin = 'hcl'
    data_A_zero = ws_load.read_ws(dout[0], kin, 'A')
    data_B_zero = ws_load.read_ws(dout[0], kin, 'B')
    ndigits_data = 4
    ndigits_model = 4
    wsa0data = pd.DataFrame()
    wsa0data['order'] = data_A_zero['order'].astype(float).round(1)
    wsa0data['wave'] = data_A_zero['wll'].astype(float).round(ndigits_data)
    wsa0data['x'] = data_A_zero['posm'].astype(float)

    wsb0data = pd.DataFrame()
    wsb0data['order'] = data_B_zero['order'].astype(float).round(1)
    wsb0data['wave'] = data_B_zero['wll'].astype(float).round(ndigits_data)
    wsb0data['x'] = data_B_zero['posm'].astype(float)

    model_A_zero = optimization.read_full_model(dout[0], kin, 'A')
    model_B_zero = optimization.read_full_model(dout[0], kin, 'B')

    wsa0model = pd.DataFrame()
    wsa0model['order'] = model_A_zero['order'].astype(float).round(1)
    wsa0model['wave'] = model_A_zero['wave'].astype(float).round(ndigits_model)
    wsa0model['x'] = model_A_zero['x'].astype(float)

    wsb0model = pd.DataFrame()
    wsb0model['order'] = model_B_zero['order'].astype(float).round(1)
    wsb0model['wave'] = model_B_zero['wave'].astype(float).round(ndigits_model)
    wsb0model['x'] = model_B_zero['x'].astype(float)

    mjd, da, da_std, db, db_std, dac, dac_std, dbc, dbc_std = [], [], [], [], [], [], [], [], []
    dd, dd_std, ddc, ddc_std = [], [], [], []
    t1, t2, t3, t4, t5, t6, t7, t8 = [], [], [], [], [], [], [], []
    pres = []
    dates = []

    #outdir = 'plots/'
    #print(spectrum)
    # fileout_dd = open(outdir + 'dd_data.csv', 'w')
    # fileout_dd.write(
    #    'mjd,d_a_mean,d_a_std,d_b_mean,d_b_std,d_a_c_mean,d_a_c_std,d_b_c_mean,d_b_c_std,dd_m_mean,dd_m_std,dd_m_w_mean,dd_c_mean,dd_c_std,dd_c_w_mean,t1,t2,t3,t4,t5,t6,t7,t8,p,date\n')

    wsdir = 'data/vis_ws_timeseries/'
    for d in dout:
        print(d)
        mjdfile = open(wsdir + d + '/mjd.dat')
        jd = mjdfile.readline()
        jd = jd.split('\'')
        jd = float(jd[1])

        mjd.append(jd)

        data_A = ws_load.read_ws(d, kin, 'A')
        data_B = ws_load.read_ws(d, kin, 'B')

        wsaidata = pd.DataFrame()
        wsaidata['order'] = data_A['order'].astype(float).round(1)
        wsaidata['wave'] = data_A['wll'].astype(float).round(ndigits_data)
        wsaidata['x'] = data_A['posm'].astype(float)

        wsbidata = pd.DataFrame()
        wsbidata['order'] = data_B['order'].astype(float).round(1)
        wsbidata['wave'] = data_B['wll'].astype(float).round(ndigits_data)
        wsbidata['x'] = data_B['posm'].astype(float)

        model_A = optimization.read_full_model(d, kin, 'A')
        model_B = optimization.read_full_model(d, kin, 'B')

        wsaimodel = pd.DataFrame()
        wsaimodel['order'] = model_A['order'].astype(float).round(1)
        wsaimodel['wave'] = model_A['wave'].astype(float).round(ndigits_model)
        wsaimodel['x'] = model_A['x'].astype(float)

        wsbimodel = pd.DataFrame()
        wsbimodel['order'] = model_B['order'].astype(float).round(1)
        wsbimodel['wave'] = model_B['wave'].astype(float).round(ndigits_model)
        wsbimodel['x'] = model_B['x'].astype(float)

        drift_data_a = pd.merge(wsa0data, wsaidata, how='inner',
                                on=['order', 'wave']).drop_duplicates()  # , suffixes=['_i', '_0'])
        drift_data_b = pd.merge(wsb0data, wsbidata, how='inner',
                                on=['order', 'wave']).drop_duplicates()  # , suffixes=['_i', '_0'])

        drift_model_a = pd.merge(wsa0model, wsaimodel, how='inner',
                                on=['order', 'wave']).drop_duplicates()  # , suffixes=['_i', '_0'])
        drift_model_b = pd.merge(wsb0model, wsbimodel, how='inner',
                                 on=['order', 'wave']).drop_duplicates()  # , suffixes=['_i', '_0'])

        # Drift A data
        d_a = drift_data_a['x_x'] - drift_data_a['x_y']
        drifta = np.median(d_a)
        da.append(drifta)
        da_std.append(np.std(d_a))
        print('Drift in A = ', drifta)

        # Drift B data
        d_b = drift_data_b['x_x'].values - drift_data_b['x_y'].values
        driftb = np.median(d_b)
        db.append(driftb)
        db_std.append(np.std(d_b))
        print('Drift in B  = ', driftb)

        # Drift A model
        drift_a = drift_model_a['x_x'].values - drift_model_a['x_y'].values
        drifta = np.median(drift_a)
        dac.append(drifta)
        dac_std.append(np.std(drift_a))
        print('Drift A model = ', drifta)

        # Drift B model
        drift_b = drift_model_b['x_x'].values - drift_model_b['x_y'].values
        driftb = np.median(drift_b)
        dbc.append(driftb)
        dbc_std.append(np.std(drift_b))
        print('Drift B model = ', driftb)

        # Differential drifts for data
        dd_data = pd.merge(wsaidata, wsbidata, how='inner', on=['order', 'wave'])
        dd_array = dd_data['x_x'].values - dd_data['x_y'].values
        dddata = np.median(dd_array)
        ddstd = np.std(dd_array)
        dd.append(dddata)
        dd_std.append(ddstd)
        print('Data differential drift = ', dddata)

        # Differential drifts for model
        dd_model = pd.merge(wsaimodel, wsbimodel, how='inner', on=['order', 'wave']).drop_duplicates()
        dd_c = dd_model['x_x'].values - dd_model['x_y'].values
        ddca = np.median(dd_c)
        ddc.append(ddca)
        ddc_std.append(np.std(dd_c))
        print('Model differential drift = ', ddca)

        # Environment data
        temps = env_data.get_t_mjd(jd)
        pressure = env_data.get_p_mjd(jd)
        t1.append(temps[0])
        t2.append(temps[1])
        t3.append(temps[2])
        t4.append(temps[3])
        t5.append(temps[4])
        t6.append(temps[5])
        t7.append(temps[6])
        t8.append(temps[7])
        pres.append(pressure)
        dates.append(d)

    out = pd.DataFrame()
    out['mjd'] = mjd
    out['da'] = da
    out['da_std'] = da_std
    out['db'] = db
    out['db_std'] = db_std
    out['dac'] = dac
    out['dac_std'] = dac_std
    out['dbc'] = dbc
    out['dbc_std'] = dbc_std
    out['dd'] = dd
    out['dd_std'] = dd_std
    out['ddc'] = ddc
    out['ddc_std'] = ddc_std
    out['t1'] = t1
    out['t2'] = t2
    out['t3'] = t3
    out['t4'] = t4
    out['t5'] = t5
    out['t6'] = t6
    out['t7'] = t7
    out['t8'] = t8
    out['pressure'] = pres
    out['date'] = dates
    out.to_csv('drifts_timeseries_v3.csv', sep=',', index=False)


def plotdd():

    data = pd.read_csv('drifts_timeseries.csv')
    plt.plot(data['mjd'], data['dd'] - np.mean(data['dd']), 'ko', alpha=0.5)
    plt.plot(data['mjd'], data['ddc'] - np.mean(data['ddc']), 'ro', alpha=0.5)
    #plt.plot(data['mjd'], data['dac'], 'b.', alpha=0.5)
    plt.show()


def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    # subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x, y, width, height])# axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize*1.5)
    subax.yaxis.set_tick_params(labelsize=y_labelsize*1.5)


    return subax


def drifts_plot():
    data = pd.read_csv('drifts_timeseries_v2.csv')
    dates = []
    for i in range(len(data)):
        d = int(data['mjd'].values[i]) + 0.5
        date = Time(d, format='jd') + 2400000.5
        date = date.isot[:10]
        dates.append(date)

    kelvin = 273.15
    fig, axes = plt.subplots(nrows=3,
                             ncols=1,
                             figsize=(14, 8),
                             gridspec_kw={'height_ratios': [1, 0.5, 0.5]},
                             sharex=True,
                             sharey=False)

    axtemp = axes[0].twinx()
    msize = 20
    legendsize = 13
    orderdata = 10
    ordermoes = 0
    axes[0].scatter(data['mjd'], -data['da'], color='r', marker='x', alpha=0.8, s=msize,
                    label=r'$d_{n, {\rm A}}$', zorder=orderdata)
    axes[0].scatter(data['mjd'], -data['db'] + 0.2, color='darkgreen', alpha=0.8, marker='x', s=msize,
                    label=r'$d_{n, {\rm B}}$ + offset', zorder=orderdata)
    axes[0].scatter(data['mjd'], -data['dac'], color='navy', alpha=0.4, marker='^', s=msize * 3.5,
                    label=r'$d_{n, \rm A, \tt moes}$', zorder=ordermoes)
    axes[0].scatter(data['mjd'], -data['dbc'] + 0.2, color='blueviolet', alpha=0.6, marker='^', s=msize * 3.5,
                    label=r'$d_{n, \rm B, \tt moes}$ + offset', zorder=ordermoes)
    # temps = (dd_data['t1'] + dd_data['t2'] + dd_data['t3'] + dd_data['t4'] + dd_data['t5'] + dd_data['t6'] + dd_data['t7'] + dd_data['t8'])/8
    axes[0].set_ylim(-0.35, 0.8)
    #axdates = axes[0].twiny()
    year_month_formatter = mdates.DateFormatter("%Y-%m")
    #axdates.xaxis.set_major_formatter(year_month_formatter)
    #axdates.plot(dates, data['da'] - 100, marker=",", c='w')

    temps = (data['t8'])
    axtemp.scatter(data['mjd'], temps + kelvin, color='k', marker='.', s=int(100), label=r'$T_i$', alpha=0.3)
    axtemp.legend(loc=4, markerscale=1, prop={'size': legendsize})
    axes[0].legend(loc=2, ncol=2, markerscale=1, prop={'size': legendsize})
    axtemp.set_ylabel(r'$T   [^\circ\rm K]$', fontsize='large', family='serif')
    axes[0].set_ylabel(r'$d_{n}$ [pix]')
    # axes[1].set_xlabel('Modified Julian Date (days)')
    axes[0].set_xlim(min(data['mjd']), max(data['mjd']))
    axtemp.set_ylim(10.1 + kelvin, 10.82 + kelvin + 0.2)
    # subaxdate = axes[0].twiny()
    # subaxdate.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    subjdini = 58550
    subjdend = 58600
    data_subset = data.loc[data['mjd'] > subjdini]
    data_subset = data_subset.loc[data_subset['mjd'] < subjdend]
    #model_subset = data.loc[data['mjd'] > subjdini]
    #model_subset = model_subset.loc[model_subset['jd'] < subjdend]
    '''
    rect = [0.53, 0.65, 0.25, 0.35]
    subax = add_subplot_axes(axes[0], rect)
    subax.scatter(data_subset['mjd'], data_subset['da'], color='r', marker='x', alpha=0.8,
                  s=msize, label=r'$d_{n, {\rm A}}$', zorder=orderdata)
    subax.scatter(data_subset['mjd'], data_subset['db'] + 0.05, color='darkgreen', alpha=0.8,
                  marker='x', s=msize, label=r'$d_{n, {\rm B}}$ + 0.25 pix', zorder=orderdata)
    subax.scatter(data_subset['mjd'], data_subset['dac'], color='navy', alpha=0.6, marker='^',
                  s=msize * 3.5, label=r'$d_{n, {\rm A}, \tt moes}$', zorder=ordermoes)
    subax.scatter(data_subset['mjd'], data_subset['dbc'] + 0.05, color='blueviolet', alpha=0.5,
                  marker='^', s=msize * 3.5, label=r'$d_{n, {\rm B}, \tt moes}$ + 0.25 pix', zorder=ordermoes)

    subaxtemp = subax.twinx()
    subaxtemp.scatter(data['mjd'], temps + 273.15, color='k', marker='.', s=int(100), label=r'T$^\circ$',
                      alpha=0.4)
    subaxtemp.set_xlim(subjdini, subjdend)
    subaxtemp.set_ylim(10.25 + 273.15 -0.1, 10.32 + 273.15 + 0.2)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subaxtemp.xaxis.set_tick_params(labelsize=x_labelsize * 1.5)
    subaxtemp.yaxis.set_tick_params(labelsize=y_labelsize * 1.5)
    '''
    axes[1].set_ylabel('O - C [pix]')
    axes[1].set_ylim(-0.005, 0.005)
    mjdaux = np.array(data['mjd'].values)
    resa = data['da'].values - data['dac'].values
    resb = data['db'].values - data['dbc'].values
    resa = np.array(resa)
    resb = np.array(resb)
    resdata = pd.DataFrame()
    resdata['mjd'] = mjdaux
    resdata['resa'] = resa
    resdata['resb'] = resb

    res_subset = resdata.loc[resdata['mjd'] > subjdini]
    res_subset = res_subset.loc[res_subset['mjd'] < subjdend]

    axes[1].scatter(mjdaux, resa, color='r', marker='x', alpha=0.8, s=msize,
                    label=r'($d_{n, {\rm A}}$ - $d_{n, {\rm A}, \tt moes}$)')
    axes[1].scatter(mjdaux, resb + 0.0, color='darkgreen', alpha=0.8, marker='x', s=msize,
                    label=r'($d_{n, {\rm B}}$ - $d_{n, {\rm B}, \tt moes}$)')
    axes[1].legend(loc=1, ncol=2, prop={'size': legendsize})
    # axes[1].set_ylim(-0.225, 0.025)
    # axes[2].tick_params(axis='x', which='minor', bottom=True)
    rms_resa = np.sqrt(np.sum(resa**2)/len(resa))
    rms_resb = np.sqrt(np.sum(resb ** 2) / len(resb))
    print(rms_resa, rms_resb)
    ticktimes = [58100, 58200, 58300, 58400, 58500, 58600]
    axes[2].set_xticks(ticktimes, minor=True)
    axes[2].scatter(data['mjd'], data['pressure'] * 1013.25, color='k', marker='.', s=int(100), label=r'Pressure',
                    alpha=0.4)
    axes[2].set_ylabel('Pressure [mbar]')
    axes[2].set_ylim(-2e-6 * 1013.25, 3e-5 * 1013.25)
    axes[2].set_xlabel('MJD [days]')
    axes[2].tick_params(axis="both", length=4, direction="in", which="both", right=True, top=False)
    axes[2].minorticks_on()

    # axes[2].set_ylim()
    # plt.plot(dd_moes['jd'], dd_moes['p'], 'ko')
    # plt.show()

    '''
    rect2 = [0.65, 0.5, 0.25, 0.35]
    subax2 = add_subplot_axes(axes[1], rect2)
    subax2.scatter(res_subset['mjd'], res_subset['resa'], color='r', marker='x', alpha=0.8, s=msize,
                    label=r'($d_{A, CAR} - d_{A, moes}$) (pix)')
    subax2.scatter(res_subset['mjd'], res_subset['resb'], color='navy', marker='x', alpha=0.8, s=msize,
                   label=r'($d_{A, CAR} - d_{A, moes}$) (pix)')
    subax2.set_ylim(-0.0075, 0.005)
    subax2.set_xlim(subjdini, subjdend)
    #subax2.
    '''
    plt.subplots_adjust(hspace=0.12)
    # plt.tight_layout()
    #plt.savefig('plots/fibers_drift_time_series.png')
    plt.show()
    plt.clf()


def ddplots():
    data = pd.read_csv('drifts_timeseries_v2.csv')
    dates = []
    for i in range(len(data)):
        d = int(data['mjd'].values[i]) + 0.5
        date = Time(d, format='jd') + 2400000.5
        date = date.isot[:10]
        dates.append(date)

    msize = 5
    #fig, axes = plt.subplots(nrows=2,
    #                         ncols=2,
    #                         figsize=(14, 5),
    #                         gridspec_kw={'height_ratios': [1, 0.5, 0.5]},
    #                         sharex=True,
    #                         sharey=False)
    # Just for the temps

    fig = plt.figure(figsize = (12, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
    axtemp = ax1.twinx()
    data['temp_grad'] = (data['t1'] + data['t2'] + data['t3'])/3. - (data['t6'] + data['t7'] + data['t8']) / 3
    data['temp_grad'] = data['temp_grad'] - np.mean(data['temp_grad'])
    data['dd_res'] = (data['dd'] - data['ddc']) - np.mean(data['dd'] - data['ddc'])
    data = data.loc[np.abs(data['temp_grad']) < 3 * np.std(data['temp_grad'])]
    data = data.loc[np.abs(data['dd_res']) < 3 * np.std(data['dd_res'])]


    #axtemp.plot(datamoes['mjd'], -datamoes['temp_grad'], 'ko', alpha=0.5, zorder=0, label=r'$\nabla T$')
    axtemp.plot(data['mjd'].values, -data['temp_grad'].values, 'ko', alpha=0.5, zorder=0, label=r'$\nabla T$')
    axtemp.legend(loc=3)
    axtemp.set_ylabel(r'$\nabla T$ [K]')
    axtemp.set_ylim(-0.01, 0.01)

    ax1.set_xlim(min(data['mjd']), max(data['mjd']))
    ax1.set_ylim(-0.0035, 0.0035)
    #ax1.errorbar(data['mjd'], data['dd_m_mean'],
    #             yerr=data['dd_m_std']/2., fmt='o', mec='cornflowerblue', ecolor='blue', elinewidth=3,
    #             mfc='white',
    #             ms=7, label=r'$\delta f_{i, data}$', zorder=10)

    ax1.plot(data['mjd'], data['dd'] - np.mean(data['dd']), 'bo', alpha=0.4, label=r'$\delta f_{n, M}$', zorder = 40)
    ax1.plot(data['mjd'], data['ddc'] - np.mean(data['ddc']), 'ro', alpha=0.4, label=r'$\delta f_{n, C}$', zorder = 31)
    ax1.set_xlabel('MJD [days]')
    #ax1.set_ylabel(r'$\delta f_n - \overline{\delta f_{n}}$ [pix]')
    ax1.set_ylabel(r'$\delta f_n$ [pix]')
    ax1.legend()
    #axdates = ax1.twiny()
    #dates = pd.to_datetime(moes['date'])

    #year_month_formatter = mdates.DateFormatter("%Y-%m")
    #axdates.xaxis.set_major_formatter(year_month_formatter)
    #axdates.plot(dates, data['dd'] - 100, marker=",", c='w')

    auxdf = pd.DataFrame()
    auxdf['x'] = data['ddc'].values - np.median(data['ddc'].values)
    auxdf['y'] = data['dd'].values - np.median(data['dd'].values)
    auxdf = auxdf.loc[auxdf['x'] < 3 * np.std(auxdf['x'])]
    auxdf = auxdf.loc[auxdf['x'] > -0.05]
    auxdf = auxdf.loc[auxdf['y'] < 3 * np.std(auxdf['y'])]

    #y = data['dd'].values - np.median(data['dd'].values)
    coef = np.polyfit(auxdf['x'], auxdf['y'], 1)
    poly1d_fn = np.poly1d(coef)
    xarray = np.arange(-10., 10.)
    ax2.plot(xarray, poly1d_fn(xarray), 'y--', zorder=10, label=r'slope = '+str(np.round(coef[0], 3)))
    ax2.plot(data['ddc'] - np.mean(data['ddc']), data['dd'] - np.mean(data['dd']),  'ko', alpha=0.5)
    ax2.set_ylabel(r'$\delta f_{n, M}$ [pix]')
    ax2.set_xlabel(r'$\delta f_{n, C}$ [pix]')
    ax2.set_ylim(-0.0038, 0.0038)
    ax2.set_xlim(-0.0038, 0.0038)
    ax2.legend(loc='best')


    #datamoes = datamoes.loc[np.abs(datamoes['temp_grad']) < 0.01]
    coef = np.polyfit(-data['temp_grad'], (data['dd'] - np.mean(data['dd'])), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    xarray = np.arange(-50, 50)
    ax3.plot(xarray, poly1d_fn(xarray), 'y--', zorder=10, label=r'slope = ' + str(np.round(coef[0], 3)) + r' pix/K')

    #datamoes = datamoes.loc[datamoes['dd_m_mean'] < 3 * np.std(datamoes['dd_m_mean'])]
    #ax3.plot(-datamoes['temp_grad'], datamoes['dd_m_mean'] - np.mean(datamoes['dd_m_mean']), 'ko', alpha=0.5)
    ax3.plot(-data['temp_grad'], data['dd'].values - np.mean(data['dd'].values), 'ko', alpha=0.5)
    ax3.set_ylabel(r'$\delta f_{n, M}$ [pix]')
    ax3.set_xlabel(r'$\nabla T$ [K]')
    ax3.set_xlim(-0.01, 0.01)
    ax3.set_ylim(-0.004, 0.004)
    ax3.legend(loc='best')
    plt.tight_layout()
    #plt.savefig('plots/dd_plot.png')
    plt.show()
    plt.clf()

    plt.figure(figsize=[10, 4])
    res = data['dd'] - data['ddc']
    rms_before = np.sqrt(np.sum(data['dd'] ** 2) / len(data))
    rms_moes = np.sqrt(np.sum(data['ddc'] ** 2) / len(data))
    rms_after = np.sqrt(np.sum((data['dd'] - data['ddc']) ** 2) / len(data))

    print(rms_before, rms_moes, rms_after)
    print('rms temp grad = ', np.sqrt(np.sum(data['temp_grad'] ** 2) / len(data)))
    plt.plot(data['mjd'], res, 'bo')
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    pcoeff2, _ = spearmanr(-data['temp_grad'], data['dd'])
    pcoeff3, _ = pearsonr(-data['temp_grad'], data['dd'])
    print(pcoeff3, pcoeff2)
    plt.show()
    plt.clf()
    pcoeff5, _ = pearsonr(data['ddc'], data['dd'])
    print(pcoeff5)


def rvs_files_dd_per_observation_final():
    # Setting directories
    out_path = 'output_files/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_path_stars = 'output_files/stars/'
    if not os.path.exists(out_path_stars):
        os.mkdir(out_path_stars)

    avc_data_path = 'caracal_rvs/CARM_VIS_AVC_201017/avcn/'
    rvc_data_path = 'caracal_rvs/CARM_VIS_RVC_201017/'

    # Creating full outfile
    datebase = '2017-10-20'
    stars_path = glob.glob(avc_data_path + 'J*.avcn.dat')
    #print(stars_path)
    # we iterate through the stars of the survey
    for i in range(len(stars_path)):
        star_id = stars_path[i][len(avc_data_path):-9]
        avc_data = pd.read_csv(stars_path[i], sep=' ',
                               names=['bjd', 'avc', 'e_avc', 'drift', 'e_drift', 'rv', 'e_rv', 'berv', 'sadrift', 'nzp',
                                      'e_nzp', 'flag_avc'])
        avc_data = avc_data.dropna()
        avc_data['bjd'] = np.around(avc_data['bjd'], decimals=4)
        if len(avc_data) > 1:

            rvc_data_per_star = pd.read_csv(rvc_data_path + str(star_id) + '/' + str(star_id) + '.rvc.dat', sep=' ',
                                            names=['bjd', 'rvc', 'e_rvc', 'drift', 'e_drift', 'rv', 'e_rv', 'berv',
                                                   'sadrift'])
            rvc_data = rvc_data_per_star.dropna()
            rvc_data['bjd'] = np.around(rvc_data['bjd'], decimals=4)
            print(star_id, np.round(i * 100 / len(stars_path), 3), '%')

            # we create output arrays
            nowsdates = []
            bjdout, rvcout, ervcout, avcout, eavcout, nzpdateout, enzpdateout = [], [], [], [], [], [], []
            nzpobsout, enzpobsout, tzppixout, tzppixwout, tzprvout, tzprvwout, rvscaleout = [], [], [], [], [], [], []
            t1out, t2out, t3out, t4out, t5out, t6out, t7out, t8out = [], [], [], [], [], [], [], []
            presout, starout, sptout = [], [], []
            ddmout, ddcout, ddmoesout = [], [], []
            etzpout, e_ddm = [], []

            for k in range(len(avc_data)):
                # Checking WS corresponding to the observation
                date_bjd = avc_data['bjd'].values[k]
                dt = (int(date_bjd) + 0.5) - date_bjd
                if dt < 0:
                    date = Time(date_bjd, format='jd').isot[:10]
                    dateaux = Time(date + 'T12:00:00.0', format='isot').jd
                    dateaux = dateaux - 1
                    date_ws = Time(dateaux, format='jd').isot[:10]
                else:
                    date = Time(date_bjd, format='jd').isot[:10]
                    date_ws = date

                rvc_data_i = rvc_data.loc[rvc_data['bjd'] == date_bjd]
                rvc = rvc_data_i['rvc'].values[0]
                e_rvc = rvc_data_i['e_rvc'].values[0]
                temps_bjd = env_data.get_temps_bjd(date_bjd)
                pressure = env_data.get_p_bjd(date_bjd)
                nzp_per_date, e_nzp_per_date = nzp(date_bjd)

                path_chromatic_a = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(date_ws) + '/chrome_coeffs_a.dat'
                path_chromatic_b = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(date_ws) + '/chrome_coeffs_b.dat'

                path_optical_a = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'seidel_coefs_a.dat'
                path_optical_b = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'seidel_coefs_b.dat'

                path_poly_a = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'poly_coefs_a.csv'
                path_poly_b = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'poly_coefs_b.csv'
                if os.path.exists(path_chromatic_a) and os.path.exists(path_chromatic_b) and os.path.exists(
                        path_optical_a) and os.path.exists(path_optical_b) and os.path.exists(path_poly_a) and os.path.exists(path_poly_b):
                    wsdir = 'data/vis_ws_timeseries/'
                    if os.path.isfile(wsdir + str(date_ws) + '/ws_hcl_A.csv') and os.path.isfile(
                            wsdir + str(date_ws) + '/ws_hcl_B.csv'):

                        jdir = wsdir + str(date_ws) + '/'
                        wsmjd = pd.read_csv(jdir + 'mjd.dat', names=['mjd'])
                        mjd_ws = wsmjd['mjd'].values[0][4:-1]
                        jd_ws = float(mjd_ws) + 2400000.5
                        #print(jd_ws)
                        #print(rvc, e_rvc)
                        drifts_data = pd.read_csv('drifts_timeseries_v2.csv')
                        date_drifts = drifts_data.loc[drifts_data['date'] == date_ws]
                        if len(date_drifts) != 0:
                            ndigits = 6
                            data_a = ws_load.read_ws(date_ws, 'hcl', 'A')
                            spec_a = ws_load.spectrum_from_data(data_a)
                            spectrum_A = pd.DataFrame()
                            spectrum_A['order'] = spec_a[:, 0]
                            spectrum_A['wave'] = spec_a[:, 1]
                            spectrum_A = spectrum_A.round(ndigits)

                            data_b = ws_load.read_ws(date_ws, 'hcl', 'B')
                            spec_b = ws_load.spectrum_from_data(data_b)
                            spectrum_B = pd.DataFrame()
                            spectrum_B['order'] = spec_b[:, 0]
                            spectrum_B['wave'] = spec_b[:, 1]
                            spectrum_B = spectrum_B.round(ndigits)

                            spec_i_dd = pd.merge(spectrum_A, spectrum_B, on=['order', 'wave'], how='inner')
                            spec = spec_i_dd.values

                            # We get datasets from merged spectrum
                            # Load parameters of the WS
                            temps_ws = env_data.get_temps_bjd(jd_ws)
                            pressure_ws = env_data.get_p_bjd(jd_ws)
                            init_state_a = parameters.load_date('A', date_ws)
                            init_state_b = parameters.load_date('B', date_ws)
                            init_state_a[-1] = pressure_ws
                            init_state_b[-1] = pressure_ws

                            # Models at the time the WS is created
                            # Fiber A
                            wsa_model_ws = vis_spectrometer.tracing(spec, init_state_a, 'A', temps_ws)
                            chromatic_coeffs_a = chromatic_aberrations.load_coeffs(date_ws, 'a')
                            chromatic_model_a = chromatic_aberrations.function(np.array(wsa_model_ws['wave'].values),
                                                                             chromatic_coeffs_a)
                            wsa_model_ws['x'] = wsa_model_ws['x'] + chromatic_model_a
                            optical_coeffs_a = optical_aberrations.load_coeffs(date_ws, 'a')
                            optical_model_a = optical_aberrations.function_seidel(wsa_model_ws, optical_coeffs_a)
                            wsa_model_ws['x'] = wsa_model_ws['x'] + optical_model_a

                            polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date_ws + '/'

                            poly_coefs_a = pd.read_csv(polydir + 'poly_coefs_a.csv', sep=',')
                            z = [poly_coefs_a['p0'].values[0],
                                 poly_coefs_a['p1'].values[0],
                                 poly_coefs_a['p2'].values[0],
                                 poly_coefs_a['p3'].values[0],
                                 poly_coefs_a['p4'].values[0]]
                            f = np.poly1d(z)
                            polymodela = f(wsa_model_ws['x'].values)
                            wsa_model_ws['x'] = wsa_model_ws['x'] + polymodela

                            # Fiber B
                            wsb_model_ws = vis_spectrometer.tracing(spec, init_state_b, 'B', temps_ws)
                            chromatic_coeffs_b = chromatic_aberrations.load_coeffs(date_ws, 'b')
                            chromatic_model_b = chromatic_aberrations.function(np.array(wsb_model_ws['wave'].values),
                                                                               chromatic_coeffs_b)
                            wsb_model_ws['x'] = wsb_model_ws['x'] + chromatic_model_b
                            optical_coeffs_b = optical_aberrations.load_coeffs(date_ws, 'b')
                            optical_model_b = optical_aberrations.function_seidel(wsb_model_ws, optical_coeffs_b)
                            wsb_model_ws['x'] = wsb_model_ws['x'] + optical_model_b

                            poly_coefs_b = pd.read_csv(polydir + 'poly_coefs_b.csv', sep=',')
                            z = [poly_coefs_b['p0'].values[0],
                                 poly_coefs_b['p1'].values[0],
                                 poly_coefs_b['p2'].values[0],
                                 poly_coefs_b['p3'].values[0],
                                 poly_coefs_b['p4'].values[0]]
                            f = np.poly1d(z)
                            polymodelb = f(wsb_model_ws['x'].values)
                            wsb_model_ws['x'] = wsb_model_ws['x'] + polymodelb

                            # Models at the time of the observation BJD
                            # Fiber A
                            init_state_a = parameters.load_date('A', date_ws)
                            init_state_a[-1] = pressure

                            wsa_model = vis_spectrometer.tracing(spec, init_state_a, 'A', temps_bjd)
                            chromatic_coeffs_a = chromatic_aberrations.load_coeffs(date_ws, 'a')
                            chromatic_model_a = chromatic_aberrations.function(np.array(wsa_model['wave'].values),
                                                                               chromatic_coeffs_a)
                            wsa_model['x'] = wsa_model['x'] + chromatic_model_a
                            optical_coeffs_a = optical_aberrations.load_coeffs(date_ws, 'a')
                            optical_model_a = optical_aberrations.function_seidel(wsa_model, optical_coeffs_a)
                            wsa_model['x'] = wsa_model['x'] + optical_model_a

                            poly_coefs_a = pd.read_csv(polydir + 'poly_coefs_a.csv', sep=',')
                            z = [poly_coefs_a['p0'].values[0],
                                 poly_coefs_a['p1'].values[0],
                                 poly_coefs_a['p2'].values[0],
                                 poly_coefs_a['p3'].values[0],
                                 poly_coefs_a['p4'].values[0]]
                            f = np.poly1d(z)
                            polymodelai = f(wsa_model['x'].values)
                            wsa_model['x'] = wsa_model['x'] + polymodelai

                            # Fiber B
                            init_state_b = parameters.load_date('B', date_ws)
                            init_state_b[-1] = pressure
                            wsb_model = vis_spectrometer.tracing(spec, init_state_b, 'B', temps_bjd)
                            chromatic_coeffs_b = chromatic_aberrations.load_coeffs(date_ws, 'b')
                            chromatic_model_b = chromatic_aberrations.function(np.array(wsb_model['wave'].values),
                                                                               chromatic_coeffs_b)
                            wsb_model['x'] = wsb_model['x'] + chromatic_model_b
                            optical_coeffs_b = optical_aberrations.load_coeffs(date_ws, 'b')
                            optical_model_b = optical_aberrations.function_seidel(wsb_model, optical_coeffs_b)
                            wsb_model['x'] = wsb_model['x'] + optical_model_b
                            poly_coefs_b = pd.read_csv(polydir + 'poly_coefs_b.csv', sep=',')
                            z = [poly_coefs_b['p0'].values[0],
                                 poly_coefs_b['p1'].values[0],
                                 poly_coefs_b['p2'].values[0],
                                 poly_coefs_b['p3'].values[0],
                                 poly_coefs_b['p4'].values[0]]
                            f = np.poly1d(z)
                            polymodelbi = f(wsb_model['x'].values)
                            wsb_model['x'] = wsb_model['x'] + polymodelbi

                            # We get rv scales
                            rvscale_a_ws, rvscale_b_ws = get_rvscale(spec, init_state_a, init_state_b, temps_ws,
                                                                     date_ws)

                            rvscale_a, rvscale_b = get_rvscale(spec, init_state_a, init_state_b, temps_bjd, date_ws)
                            #e_rvscale = np.std(rvscale_a)
                            rvscale_ws = (rvscale_a_ws + rvscale_b_ws) / 2
                            rvscale_obs = (rvscale_a + rvscale_b) / 2

                            if not math.isnan(np.mean(rvscale_ws)) and not math.isnan(np.mean(rvscale_obs)):
                                rvscale_ws_hist, bins_hist_ws = np.histogram(rvscale_ws, bins=30)
                                rvscale_obs_hist, bins_hist_obs = np.histogram(rvscale_obs, bins=30)
                                bins_hist_ws = bins_hist_ws[1:] - (bins_hist_ws[2] - bins_hist_ws[1]) / 2
                                bins_hist_obs = bins_hist_obs[1:] - (bins_hist_obs[2] - bins_hist_obs[1]) / 2

                                rvscale_ws_weighted_mean = np.sum(bins_hist_ws * (rvscale_ws_hist / len(rvscale_ws)))
                                rvscale_obs_weighted_mean = np.sum(
                                    bins_hist_obs * (rvscale_obs_hist / len(rvscale_obs)))
                                # We calculate the differential drift at the WS and observing time, respectively

                                #print('RV scales = ', np.mean(rvscale_ws_weighted_mean), np.mean(rvscale_obs_weighted_mean))
                                dd_c_moes_ws = drifts_data['ddc'].values[0]
                                dd_m_data_ws = drifts_data['dd'].values[0]
                                #print('DD moes at ws = ', dd_c_moes_ws)
                                #print('DD data at ws = ', dd_m_data_ws)
                                dd_c_moes_obs = wsa_model['x'].values - wsb_model['x'].values
                                #print('DD moes at bjd = ', np.mean(dd_c_moes_obs))

                                tzp_test = (dd_m_data_ws + (np.mean(dd_c_moes_obs) - dd_c_moes_ws)) * rvscale_obs_weighted_mean
                                #print('TZP value = ', tzp_test, ' m/s')
                                # Moe's DD distribution for the WS and for the observation
                                tzp_mean = (np.mean(dd_c_moes_obs) - np.mean(dd_c_moes_ws)) * (
                                            rvscale_obs_weighted_mean + rvscale_ws_weighted_mean) / 2

                                tzp_pix = np.mean(dd_c_moes_obs - dd_c_moes_ws)
                                rvscale_mean = (rvscale_ws_weighted_mean + rvscale_obs_weighted_mean) / 2
                                tzp_rv = np.mean(
                                    (drifts_data['dd'].values[0] + dd_c_moes_obs - dd_c_moes_ws) * rvscale_mean)
                                e_tzp_rv = np.std(
                                    (drifts_data['dd'].values[0] + dd_c_moes_obs - dd_c_moes_ws) * rvscale_mean)

                                if not math.isnan(tzp_mean):
                                    #print('TZP = ', tzp_rv)
                                    # Spectral type
                                    spt = get_spec_type(star_id)
                                    # print(spt)
                                    sptout.append(spt)
                                    bjdout.append(date_bjd)
                                    rvcout.append(rvc)
                                    ervcout.append(e_rvc)
                                    avcout.append(avc_data['avc'].values[k])
                                    eavcout.append(avc_data['e_avc'].values[k])
                                    nzpdateout.append(nzp_per_date)
                                    enzpdateout.append(e_nzp_per_date)
                                    nzpobsout.append(avc_data['nzp'].values[k])
                                    enzpobsout.append(avc_data['e_nzp'].values[k])
                                    ddmout.append(drifts_data['dd'].values[0])
                                    ddmoesout.append(np.mean(dd_c_moes_obs))
                                    ddcout.append(drifts_data['ddc'].values[0])
                                    t1out.append(temps_bjd[1])
                                    t2out.append(temps_bjd[2])
                                    t3out.append(temps_bjd[3])
                                    t4out.append(temps_bjd[4])
                                    t5out.append(temps_bjd[5])
                                    t6out.append(temps_bjd[6])
                                    t7out.append(temps_bjd[7])
                                    t8out.append(temps_bjd[8])
                                    presout.append(pressure)
                                    starout.append(star_id)
                                    tzppixout.append(tzp_pix)
                                    rvscaleout.append(rvscale_mean)
                                    tzprvout.append(tzp_test)
                                    etzpout.append(e_tzp_rv)
                                    e_ddm.append(drifts_data['dd_std'].values[0] * rvscale_mean)

                                else:
                                    print('Files corrupted...')
                        else:
                            nowsdates.append(date)

                else:
                    print('No coefficients for date ', date_ws)

            outdata = pd.DataFrame()
            outdata['bjd'] = bjdout
            outdata['rvc'] = rvcout
            outdata['e_rvc'] = ervcout
            outdata['avc'] = avcout
            outdata['e_avc'] = eavcout
            outdata['nzp_date'] = nzpdateout
            outdata['e_nzp_date'] = enzpdateout
            outdata['nzp'] = nzpobsout
            outdata['e_nzp'] = enzpobsout
            outdata['tzp_pix'] = tzppixout
            outdata['tzp'] = tzprvout
            outdata['e_tzp'] = etzpout
            outdata['dd_m_date'] = ddmout
            outdata['e_dd_m_date'] = e_ddm
            outdata['dd_c_date'] = ddcout
            outdata['dd_moes_obs'] = ddmoesout
            outdata['pix2ms'] = rvscaleout
            outdata['t1'] = t1out
            outdata['t2'] = t2out
            outdata['t3'] = t3out
            outdata['t4'] = t4out
            outdata['t5'] = t5out
            outdata['t6'] = t6out
            outdata['t7'] = t7out
            outdata['t8'] = t8out
            outdata['pressure'] = presout
            outdata['starid'] = starout
            outdata['spt'] = sptout
            outdata.to_csv('data/stars/' + str(star_id) + '_avc_dd.csv', index=False, sep=',')
            print('File of star ', star_id, ' written')
            # stars_done.write(star_id+'\n')


def rvs_files_dd_per_observation_final_v2():
    # Setting directories

    avc_data_path = 'caracal_rvs/CARM_VIS_AVC_201017/avcn/'
    rvc_data_path = 'caracal_rvs/CARM_VIS_RVC_201017/'

    # Creating full outfile
    # datebase = '2017-10-20'
    stars_path = glob.glob(avc_data_path + 'J*.avcn.dat')
    #print(stars_path)
    stars_path = stars_path[205:]
    print(stars_path)

    # we iterate through the stars of the survey
    for i in range(len(stars_path)):
        star_id = stars_path[i][len(avc_data_path):-9]
        avc_data = pd.read_csv(stars_path[i], sep=' ',
                               names=['bjd', 'avc', 'e_avc', 'drift', 'e_drift', 'rv', 'e_rv', 'berv', 'sadrift', 'nzp',
                                      'e_nzp', 'flag_avc'])
        avc_data = avc_data.dropna()
        avc_data['bjd'] = np.around(avc_data['bjd'], decimals=4)
        if len(avc_data) > 1:

            rvc_data_per_star = pd.read_csv(rvc_data_path + str(star_id) + '/' + str(star_id) + '.rvc.dat', sep=' ',
                                            names=['bjd', 'rvc', 'e_rvc', 'drift', 'e_drift', 'rv', 'e_rv', 'berv',
                                                   'sadrift'])
            rvc_data = rvc_data_per_star.dropna()
            rvc_data['bjd'] = np.around(rvc_data['bjd'], decimals=4)
            print(star_id, np.round(i * 100 / len(stars_path), 3), '%')

            # we create output arrays
            nowsdates = []
            bjdout, rvcout, ervcout, avcout, eavcout, nzpdateout, enzpdateout = [], [], [], [], [], [], []
            nzpobsout, enzpobsout, tzppixout, tzppixwout, tzprvout, tzprvwout, rvscaleout = [], [], [], [], [], [], []
            t1out, t2out, t3out, t4out, t5out, t6out, t7out, t8out = [], [], [], [], [], [], [], []
            presout, starout, sptout = [], [], []
            ddmout, ddcout, ddmoesout = [], [], []
            etzpout, e_ddm = [], []
            tzp1, tzp2, tzp3, tzp4 = [], [], [], []
            for k in range(len(avc_data)):
                # Checking WS corresponding to the observation
                date_bjd = avc_data['bjd'].values[k]
                dt = (int(date_bjd) + 0.5) - date_bjd
                if dt < 0:
                    date = Time(date_bjd, format='jd').isot[:10]
                    dateaux = Time(date + 'T12:00:00.0', format='isot').jd
                    dateaux = dateaux - 1
                    date_ws = Time(dateaux, format='jd').isot[:10]
                else:
                    date = Time(date_bjd, format='jd').isot[:10]
                    date_ws = date

                rvc_data_i = rvc_data.loc[rvc_data['bjd'] == date_bjd]
                rvc = rvc_data_i['rvc'].values[0]
                e_rvc = rvc_data_i['e_rvc'].values[0]
                temps_bjd = env_data.get_temps_bjd(date_bjd)
                pressure = env_data.get_p_bjd(date_bjd)
                nzp_per_date, e_nzp_per_date = nzp(date_bjd)

                path_chromatic_a = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(date_ws) + '/chrome_coeffs_a.dat'
                path_chromatic_b = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(date_ws) + '/chrome_coeffs_b.dat'

                path_optical_a = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'seidel_coefs_a.dat'
                path_optical_b = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'seidel_coefs_b.dat'

                path_poly_a = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'poly_coefs_a.csv'
                path_poly_b = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'poly_coefs_b.csv'
                if os.path.exists(path_chromatic_a) and os.path.exists(path_chromatic_b) and os.path.exists(
                        path_optical_a) and os.path.exists(path_optical_b) and os.path.exists(path_poly_a) and os.path.exists(path_poly_b):
                    wsdir = 'data/vis_ws_timeseries/'
                    if os.path.isfile(wsdir + str(date_ws) + '/ws_hcl_A.csv') and os.path.isfile(
                            wsdir + str(date_ws) + '/ws_hcl_B.csv'):

                        jdir = wsdir + str(date_ws) + '/'
                        wsmjd = pd.read_csv(jdir + 'mjd.dat', names=['mjd'])
                        mjd_ws = wsmjd['mjd'].values[0][4:-1]
                        jd_ws = float(mjd_ws) + 2400000.5
                        kin = 'hcl'
                        # We load data
                        data_A = ws_load.read_ws(date_ws, kin, 'A')
                        data_B = ws_load.read_ws(date_ws, kin, 'B')

                        # We compute differential drifts
                        ndigits_data = 6
                        wsaidata = pd.DataFrame()
                        wsaidata['order'] = data_A['order'].astype(float).round(1)
                        wsaidata['wave'] = data_A['wll'].astype(float).round(ndigits_data)
                        wsaidata['wavec'] = data_A['wlc'].astype(float).round(ndigits_data)
                        wsaidata['x'] = data_A['posm'].astype(float)

                        wsbidata = pd.DataFrame()
                        wsbidata['order'] = data_B['order'].astype(float).round(1)
                        wsbidata['wave'] = data_B['wll'].astype(float).round(ndigits_data)
                        wsaidata['wavec'] = data_B['wlc'].astype(float).round(ndigits_data)
                        wsbidata['x'] = data_B['posm'].astype(float)

                        dd_data = pd.merge(wsaidata, wsbidata, how='inner', on=['order', 'wave']).drop_duplicates()
                        dd_array = dd_data['x_x'].values - dd_data['x_y'].values

                        specdd_data = pd.DataFrame()
                        specdd_data['order'] = dd_data['order'].values
                        specdd_data['wll'] = dd_data['wave'].values

                        # We get spectra to model
                        spec = np.array([dd_data['order'].values, dd_data['wavec'].values.astype(float)*1e-4]).T

                        # We compute models at WS time creation
                        temps_ws = env_data.get_temps_bjd(jd_ws)
                        pressure_ws = env_data.get_p_bjd(jd_ws)
                        init_state_a = parameters.load_date('A', date_ws)
                        init_state_b = parameters.load_date('B', date_ws)
                        init_state_a[-1] = pressure_ws
                        init_state_b[-1] = pressure_ws

                        wsa_model_ws = vis_spectrometer.tracing(spec, init_state_a, 'A', temps_ws)
                        chromatic_coeffs_a = chromatic_aberrations.load_coeffs(date_ws, 'a')
                        chromatic_model_a = chromatic_aberrations.function(np.array(wsa_model_ws['wave'].values),
                                                                           chromatic_coeffs_a)
                        wsa_model_ws['x'] = wsa_model_ws['x'] + chromatic_model_a
                        optical_coeffs_a = optical_aberrations.load_coeffs(date_ws, 'a')
                        optical_model_a = optical_aberrations.function_seidel(wsa_model_ws, optical_coeffs_a)
                        wsa_model_ws['x'] = wsa_model_ws['x'] + optical_model_a

                        polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date_ws + '/'

                        poly_coefs_a = pd.read_csv(polydir + 'poly_coefs_a.csv', sep=',')
                        z = [poly_coefs_a['p0'].values[0],
                             poly_coefs_a['p1'].values[0],
                             poly_coefs_a['p2'].values[0],
                             poly_coefs_a['p3'].values[0],
                             poly_coefs_a['p4'].values[0]]
                        f = np.poly1d(z)
                        polymodela = f(wsa_model_ws['x'].values)
                        wsa_model_ws['x'] = wsa_model_ws['x'] + polymodela

                        # Fiber B
                        wsb_model_ws = vis_spectrometer.tracing(spec, init_state_b, 'B', temps_ws)
                        chromatic_coeffs_b = chromatic_aberrations.load_coeffs(date_ws, 'b')
                        chromatic_model_b = chromatic_aberrations.function(np.array(wsb_model_ws['wave'].values),
                                                                           chromatic_coeffs_b)
                        wsb_model_ws['x'] = wsb_model_ws['x'] + chromatic_model_b
                        optical_coeffs_b = optical_aberrations.load_coeffs(date_ws, 'b')
                        optical_model_b = optical_aberrations.function_seidel(wsb_model_ws, optical_coeffs_b)
                        wsb_model_ws['x'] = wsb_model_ws['x'] + optical_model_b

                        poly_coefs_b = pd.read_csv(polydir + 'poly_coefs_b.csv', sep=',')
                        z = [poly_coefs_b['p0'].values[0],
                             poly_coefs_b['p1'].values[0],
                             poly_coefs_b['p2'].values[0],
                             poly_coefs_b['p3'].values[0],
                             poly_coefs_b['p4'].values[0]]
                        f = np.poly1d(z)
                        polymodelb = f(wsb_model_ws['x'].values)
                        wsb_model_ws['x'] = wsb_model_ws['x'] + polymodelb

                        ## End of computing models at ws time

                        # Differential drift
                        dd_model_ws = wsa_model_ws['x'].values - wsb_model_ws['x'].values

                        # DD model at BJD

                        init_state_a = parameters.load_date('A', date_ws)
                        init_state_b = parameters.load_date('B', date_ws)
                        init_state_a[-1] = pressure
                        init_state_b[-1] = pressure

                        wsa_model_bjd = vis_spectrometer.tracing(spec, init_state_a, 'A', temps_bjd)
                        chromatic_coeffs_a = chromatic_aberrations.load_coeffs(date_ws, 'a')
                        chromatic_model_a = chromatic_aberrations.function(
                            np.array(wsa_model_bjd['wave'].values),
                            chromatic_coeffs_a)
                        wsa_model_bjd['x'] = wsa_model_bjd['x'] + chromatic_model_a
                        optical_coeffs_a = optical_aberrations.load_coeffs(date_ws, 'a')
                        optical_model_a = optical_aberrations.function_seidel(wsa_model_bjd, optical_coeffs_a)
                        wsa_model_bjd['x'] = wsa_model_bjd['x'] + optical_model_a

                        polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date_ws + '/'

                        poly_coefs_a = pd.read_csv(polydir + 'poly_coefs_a.csv', sep=',')
                        z = [poly_coefs_a['p0'].values[0],
                             poly_coefs_a['p1'].values[0],
                             poly_coefs_a['p2'].values[0],
                             poly_coefs_a['p3'].values[0],
                             poly_coefs_a['p4'].values[0]]
                        f = np.poly1d(z)
                        polymodela = f(wsa_model_bjd['x'].values)
                        wsa_model_bjd['x'] = wsa_model_bjd['x'] + polymodela

                        # Fiber B
                        wsb_model_bjd = vis_spectrometer.tracing(spec, init_state_b, 'B', temps_bjd)
                        chromatic_coeffs_b = chromatic_aberrations.load_coeffs(date_ws, 'b')
                        chromatic_model_b = chromatic_aberrations.function(
                            np.array(wsb_model_bjd['wave'].values),
                            chromatic_coeffs_b)
                        wsb_model_bjd['x'] = wsb_model_bjd['x'] + chromatic_model_b
                        optical_coeffs_b = optical_aberrations.load_coeffs(date_ws, 'b')
                        optical_model_b = optical_aberrations.function_seidel(wsb_model_bjd, optical_coeffs_b)
                        wsb_model_bjd['x'] = wsb_model_bjd['x'] + optical_model_b

                        polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date_ws + '/'

                        poly_coefs_b = pd.read_csv(polydir + 'poly_coefs_b.csv', sep=',')
                        z = [poly_coefs_b['p0'].values[0],
                             poly_coefs_b['p1'].values[0],
                             poly_coefs_b['p2'].values[0],
                             poly_coefs_b['p3'].values[0],
                             poly_coefs_b['p4'].values[0]]
                        f = np.poly1d(z)
                        polymodelb = f(wsb_model_bjd['x'].values)
                        wsb_model_bjd['x'] = wsb_model_bjd['x'] + polymodelb

                        dd_bjd = wsa_model_bjd['x'].values - wsb_model_bjd['x'].values
                        # We get rv scales
                        rvscale_a_ws, rvscale_b_ws = get_rvscale(spec, init_state_a, init_state_b, temps_ws,
                                                                 date_ws)

                        rvscale_a, rvscale_b = get_rvscale(spec, init_state_a, init_state_b, temps_bjd, date_ws)

                        rvscale_ws = (rvscale_a_ws + rvscale_b_ws) / 2
                        rvscale_obs = (rvscale_a + rvscale_b) / 2

                        tzparray = pd.DataFrame()
                        tzparray['dd_m'] = dd_array
                        tzparray['dd_c'] = dd_model_ws
                        tzparray['dd_c_obs'] = dd_bjd
                        tzparray['rvscale_ws'] = rvscale_ws
                        tzparray['rvscale_obs'] = rvscale_obs
                        tzparray = tzparray.dropna()
                        if len(tzparray) > 0:
                            ddc_ws_median, ddc_ws_lo, ddc_ws_hi = get_sum(tzparray['dd_c'].values)
                            dd_obs_median, dd_obs_lo, dd_obs_hi = get_sum(tzparray['dd_c_obs'].values)
                            dd_m_median, dd_m_lo, dd_m_hi = get_sum(tzparray['dd_m'].values)

                            # dd_c_diff = 2*tzparray['dd_c'].values - tzparray['dd_c_obs'].values
                            tzp_pix = tzparray['dd_m'].values + (tzparray['dd_c'].values - tzparray['dd_c_obs'].values)

                            # tzp_c = (2*tzparray['dd_c'].values - tzparray['dd_c_obs'].values) * tzparray['rvscale_obs'].values
                            tzp_m = (tzparray['dd_m'].values + tzparray['dd_c'].values - tzparray['dd_c_obs'].values) * \
                                    tzparray['rvscale_obs'].values

                            rvscale, rvscale_lo, rvscale_hi = get_sum(tzparray['rvscale_ws'].values)

                            # plt.hist(tzparray['dd_m'].values, bins =30, alpha=0.5)
                            # plt.hist(tzparray['dd_c'].values, bins=30, alpha=0.5, color='red')
                            # plt.hist(tzparray['dd_c_obs'].values, bins=30, alpha=0.5, color='purple')
                            # plt.hist(tzp_c, bins=40, alpha=0.5, color='red')
                            # plt.hist(tzp_m, bins=200, alpha=0.5, color='purple')
                            # plt.hist(tzparray['rvscale_ws'].values, bins=50, alpha=0.5, color='red')
                            # plt.hist(tzparray['rvscale_obs'].values, bins=50, alpha=0.5, color='purple')

                            tzp_m_median, tzp_m_lo, tzp_m_hi = get_sum(tzp_m)
                            nc, binsc, patchc = plt.hist(tzp_m, color='b', alpha=0.4, bins=200, edgecolor='black')
                            poptc, pcovc = curve_fit(gaus, (binsc[1:] - (binsc[1] - binsc[0]) / 2), nc,
                                                     p0=[max(nc), tzp_m_median, tzp_m_lo])
                            # xrange = np.arange(-200., 400., 1.)
                            # plt.plot(xrange, gaus(xrange, *poptc), 'k-')
                            # plt.show()

                            tzp_gauss = poptc[1]
                            tzp_w_hist, bins_hist = np.histogram(tzp_m, bins=150)
                            bins_hist = bins_hist[1:] - (bins_hist[2] - bins_hist[1]) / 2
                            tzp_weight = np.sum(bins_hist * (tzp_w_hist / len(tzp_m)))

                            tzp_mean = (np.mean(tzparray['dd_m'].values) + np.mean(tzparray['dd_c'].values) - np.mean(
                                tzparray['dd_c_obs'].values)) * np.mean(tzparray['rvscale_obs'].values)
                            if not math.isnan(tzp_mean):
                                # print('TZP = ', tzp_rv)
                                # Spectral type
                                spt = get_spec_type(star_id)
                                # print(spt)
                                sptout.append(spt)
                                bjdout.append(date_bjd)
                                rvcout.append(rvc)
                                ervcout.append(e_rvc)
                                avcout.append(avc_data['avc'].values[k])
                                eavcout.append(avc_data['e_avc'].values[k])
                                nzpdateout.append(nzp_per_date)
                                enzpdateout.append(e_nzp_per_date)
                                nzpobsout.append(avc_data['nzp'].values[k])
                                enzpobsout.append(avc_data['e_nzp'].values[k])
                                ddmout.append(dd_m_median)
                                ddmoesout.append(ddc_ws_median)
                                ddcout.append(dd_obs_median)
                                t1out.append(temps_bjd[1])
                                t2out.append(temps_bjd[2])
                                t3out.append(temps_bjd[3])
                                t4out.append(temps_bjd[4])
                                t5out.append(temps_bjd[5])
                                t6out.append(temps_bjd[6])
                                t7out.append(temps_bjd[7])
                                t8out.append(temps_bjd[8])
                                presout.append(pressure)
                                starout.append(star_id)
                                tzp1.append(tzp_m_median)
                                tzp2.append(tzp_gauss)
                                tzp3.append(tzp_weight)
                                tzp4.append(tzp_mean)
                                tzppixout.append(tzp_pix)
                                rvscaleout.append(rvscale)
                                print(k, ' TZP, median = ', tzp_m_median, ', gaussian fit = ', tzp_gauss,
                                      ', weight hist = ',
                                      tzp_weight, ', mean = ', tzp_mean)

                        else:
                            print('Files corrupted...')
                    else:
                        nowsdates.append(date)

                else:
                    print('No coefficients for date ', date_ws)

            outdata = pd.DataFrame()
            outdata['bjd'] = bjdout
            outdata['rvc'] = rvcout
            outdata['e_rvc'] = ervcout
            outdata['avc'] = avcout
            outdata['e_avc'] = eavcout
            outdata['nzp_date'] = nzpdateout
            outdata['e_nzp_date'] = enzpdateout
            outdata['nzp'] = nzpobsout
            outdata['e_nzp'] = enzpobsout
            outdata['tzp_pix'] = tzppixout
            outdata['tzp_median'] = tzp1
            outdata['tzp_gauss'] = tzp2
            outdata['tzp_weighted'] = tzp3
            outdata['tzp_mean'] = tzp4
            outdata['dd_m_date'] = ddmout
            outdata['dd_moes_date'] = ddcout
            outdata['dd_moes_obs'] = ddmoesout
            outdata['pix2ms'] = rvscaleout
            outdata['t1'] = t1out
            outdata['t2'] = t2out
            outdata['t3'] = t3out
            outdata['t4'] = t4out
            outdata['t5'] = t5out
            outdata['t6'] = t6out
            outdata['t7'] = t7out
            outdata['t8'] = t8out
            outdata['pressure'] = presout
            outdata['starid'] = starout
            outdata['spt'] = sptout
            outdata.to_csv('data/stars/' + str(star_id) + '_avc_dd.csv', index=False, sep=',')
            print('File of star ', star_id, ' written')
            # stars_done.write(star_id+'\n')


def tvc_all_stars():

    tvcpath = 'data/stars/'
    tvcfiles = glob.glob(tvcpath + '*')
    bjd, nzp, tzp, rvc, avc = [], [], [], [], []
    tzp_med, tzp_gauss, tzp_weight, tzp_mean = [], [], [], []
    t1, t2, t3, t4, t5, t6, t7, t8 = [], [], [], [], [], [], [], []
    starid = []
    for f in tvcfiles:

        data = pd.read_csv(f, sep=',')
        for k in range(len(data)):
            if np.abs(data['nzp'].values[k]) < 5.:
                idstar = f[11:-11]
                starid.append(idstar)
                bjd.append(data['bjd'].values[k])
                rvc.append(data['rvc'].values[k])
                avc.append(data['avc'].values[k])
                nzp.append(data['nzp'].values[k])

                # tzp.append(data['tzp'].values[k])
                # dd_m.append(data['dd_m_date'].values[k])
                # dd_c.append(data['dd_c_date'].values[k])
                # dd_obs.append(data['dd_moes_obs'].values[k])
                tzp_med.append(data['tzp_median'].values[k])
                tzp_gauss.append(data['tzp_gauss'].values[k])
                tzp_weight.append(data['tzp_weighted'].values[k])
                tzp_mean.append(data['tzp_mean'].values[k])
                t1.append(data['t1'].values[k])
                t2.append(data['t2'].values[k])
                t3.append(data['t3'].values[k])
                t4.append(data['t4'].values[k])
                t5.append(data['t5'].values[k])
                t6.append(data['t6'].values[k])
                t7.append(data['t7'].values[k])
                t8.append(data['t8'].values[k])

            #rvscale.append(data['pix2ms'].values[k] * data['dd_c_date'].values[k] + data['tzp'].values[k])

    #rvscale = np.array(rvscale)
    nzp = np.array(nzp)
    bjd = np.array(bjd)
    tzp1 = np.array(tzp_med)# - np.mean(np.array(tzp_med))
    tzp2 = np.array(tzp_gauss)# - np.mean(np.array(tzp_gauss))
    tzp3 = np.array(tzp_weight)# - np.mean(np.array(tzp_weight))
    tzp4 = np.array(tzp_mean) #- np.mean(np.array(tzp_mean))
    plt.figure(figsize=[8, 3])
    plt.plot(bjd, nzp, 'k.', alpha=0.5)
    plt.plot(bjd, -np.array(tzp1), 'r.', alpha=0.5)
    # plt.xlabel('BJD (days)')
    # plt.savefig('plot_nzp_test.png')
    plt.show()
    plt.clf()
    plt.close()

    plt.plot(-np.array(tzp1), nzp, 'k.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.show()

    def line(x, a, b):
        return a * x + b

    # plt.figure(figsize=[5,5])
    # plt.plot(, dd_m, 'r.', alpha=0.5)
    a = 0.
    b = 0.
    # popt, pcov = curve_fit(line, dd_c, nzp)
    # xrange = np.arange(min(dd_c), max(dd_c), 0.001)
    # plt.plot(dd_c, nzp, 'b.', alpha=0.5)
    # plt.plot(xrange, line(xrange, *popt), 'k-')
    # plt.plot(bjd, dd_obs, 'g.', alpha=0.5)
    # plt.savefig('plot_dd_test.png')
    # plt.show()
    plt.clf()

    all_data = pd.DataFrame()
    all_data['bjd'] = bjd
    all_data['nzp'] = nzp
    all_data['rvc'] = rvc
    all_data['avc'] = avc
    all_data['tzp_med'] = tzp1
    all_data['tzp_gauss'] = tzp2
    all_data['tzp_weight'] = tzp3
    all_data['tzp_mean'] = tzp4
    all_data['t1'] = np.array(t1)
    all_data['t2'] = np.array(t2)
    all_data['t3'] = np.array(t3)
    all_data['t4'] = np.array(t4)
    all_data['t5'] = np.array(t5)
    all_data['t6'] = np.array(t6)
    all_data['t7'] = np.array(t7)
    all_data['t8'] = np.array(t8)
    all_data['id'] = np.array(starid)
    #all_data['avg_temp'] = (all_data['t1'] + all_data['t2'] + all_data['t3'] + all_data['t4'] + all_data['t5'] +
    #                        all_data['t6'] + all_data['t7'] + all_data['t8']) / 8
    #all_data['temp_grad_t2t7'] = all_data['t2'] - all_data['t7']
    #all_data['temp_grad_t1t8'] = all_data['t1'] - all_data['t8']
    #all_data['temp_grad_t1t2'] = all_data['t1'] - all_data['t2']
    #all_data['temp_grad_t8t4'] = all_data['t8'] - all_data['t4']
    all_data.to_csv('data/all_stars_tzp_full.csv', index=False)
    #mean_tg_1_8 = np.mean(all_data['temp_grad_t1t8'].values)
    #std_tg_1_8 = np.std(all_data['temp_grad_t1t8'].values)

    #temp_grad_data = all_data.loc[all_data['temp_grad_t1t8'] < mean_tg_1_8 + 3 * std_tg_1_8]
    #temp_grad_data = temp_grad_data.loc[temp_grad_data['temp_grad_t1t8'] > mean_tg_1_8 - 3 * std_tg_1_8]
    # temp_grad_data = temp_grad_data.loc[temp_grad_data['temp_grad_t1t8'] < 0.51]

    #mean_res_dd_rv = np.mean(temp_grad_data['res_dd_rv_mean'].values)
    #std_res_dd_rv = np.std(temp_grad_data['res_dd_rv_mean'].values)
    #temp_grad_data = temp_grad_data.loc[temp_grad_data['res_dd_rv_mean'] < mean_res_dd_rv + 3 * std_res_dd_rv]
    #temp_grad_data = temp_grad_data.loc[temp_grad_data['res_dd_rv_mean'] > mean_res_dd_rv - 3 * std_res_dd_rv]

    #mean_nzp_obs = np.mean(temp_grad_data['nzp_obs'].values)
    #std_nzp_obs = np.std(temp_grad_data['nzp_obs'].values)
    #temp_grad_data = temp_grad_data.loc[temp_grad_data['nzp_obs'] < mean_nzp_obs + 3 * std_nzp_obs]
    #temp_grad_data = temp_grad_data.loc[temp_grad_data['nzp_obs'] > mean_nzp_obs - 3 * std_nzp_obs]


def tzp_vs_nzp_plot():
    data = pd.read_csv('data/all_stars_tzp.csv', sep=',')
    #data = pd.read_csv('all_stars_tzp_filtered.csv', sep=',')
    spout = []
    data['tzp_med'] = -(data['tzp_med'] - np.median(data['tzp_med']))
    #data = data.loc[np.abs(data['nzp']) < 4.5]
    #data = data.loc[np.abs(data['tzp_med']) < 4.5]
    #data['temp_grad'] = (data['t1'] + data['t2'] + data['t3']) / 3 - (data['t6'] + data['t7'] + data['t8']) / 3
    #data['temp_grad'] = data['temp_grad'] - np.median(data['temp_grad'])
    #for i in range(len(data)):
    #    spstr = data['spt'].values[i]
    #    spno = float(str(spstr[3:-4]))
    #    spout.append(spno)
    #print(sptzp)
    #spout = np.array(spout)/max(spout)
    #norm = mpl.colors.Normalize(vmin=0, vmax=1)
    #sm = plt.cm.ScalarMappable(cmap = 'brg', norm=norm)
    #sm.set_array(spout)
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
    axtemp = ax1.twinx()
    #divider = make_axes_locatable(ax1)
    #cax = divider.new_vertical( size = '5%', pad=0.5)
    #plt.add_axes(cax)
    #plt.colorbar(spout, cax = cax, orientation='horizontal')

    #print(data['tzp_med'])
    dt = (data['t1'] + data['t2'] + data['t3']) / 3 - (data['t6'] + data['t7'] + data['t8']) / 3
    dtarray = pd.DataFrame()
    dtarray['bjd'] = data['bjd'].values
    dtarray['dt'] = dt - np.mean(dt)
    dtaux = dtarray.loc[dtarray['bjd'] <= 2458497]
    dtaux2 = dtarray.loc[dtarray['bjd'] >= 2458500]
    dttoconv = [dtaux, dtaux2]
    dtarr = pd.concat(dttoconv)
    #print(dtarr)
    dtout, nzpout, bjdout = [], [], []
    for i in range(len(dtarr)):
        datum = data.loc[data['bjd'] == dtarr['bjd'].values[i]]
        bjdout.append(datum['bjd'].values[0])
        dtout.append(dtarr['dt'].values[i])
        nzpout.append(datum['nzp'].values[0])

    #dtarray = dtarray.loc[dtarray['bjd'] >= 2458500]
    ax1.set_xlim(min(data['bjd']) - 2458000, max(data['bjd']) - 2458000)
    ax1.set_ylim(-7, 7)
    #spout = np.full(len(spout), 0.2)
    #ax1.errorbar(data['bjd'], data['nzp'], yerr=data['e_nzp'], fmt='.', alpha=0.5, label='NZP', zorder=1)
    ax1.plot(data['bjd'] - 2458000, data['nzp'], 'bo', alpha=0.4, label=r'NZP', zorder = 20, ms = 3)
    ax1.plot(data['bjd'] - 2458000, data['tzp_med'], 'ro', alpha=0.4, label=r'TZP', zorder = 40, ms = 3)
    ax1.set_xlabel('BJD - 2458000 [days]')
    ax1.set_ylabel('RV zero points [m/s]')
    ax1.legend(loc = 1, markerscale = 3)
    axtemp.plot(dtaux['bjd'] - 2458000, dtaux['dt'], 'k.', alpha=0.5, zorder=0, label=r'$\nabla T$')
#    axtemp.plot(data['bjd'], data['temp_grad'].values, 'k.', alpha=0.5, zorder=0, label=r'$\nabla T$')
    axtemp.plot(dtaux2['bjd'] - 2458000, dtaux2['dt'], 'k.', alpha=0.5, zorder=0)
    axtemp.legend(loc=2, markerscale=3)
    axtemp.set_ylabel(r'$\nabla T$ [K]')
   # axtemp.set_ylim(-0.01, 0.01)
    ax1.set_zorder(axtemp.get_zorder() + 1)
    ax1.set_frame_on(False)

    bjds = data['bjd'].astype(int)
    bjds = np.array(bjds)
    dates = Time(bjds, format='jd').isot
    datesaux = []
    for date in dates:
        datesaux.append(date[:-13])

    axdates = ax1.twiny()
    dates = pd.to_datetime(datesaux)

    year_month_formatter = mdates.DateFormatter("%Y-%m")
    axdates.xaxis.set_major_formatter(year_month_formatter)
    axdates.plot(dates, data['nzp']-100, marker=".", c='w')

    #axdates = ax1.twiny()
    #dates = pd.to_datetime(moes['date'])
    #year_month_formatter = mdates.DateFormatter("%Y-%m")
    #axdates.xaxis.set_major_formatter(year_month_formatter)
    #axdates.plot(dates, moes['dd'] - 100, marker=",", c='w')
    coef = np.polyfit(data['tzp_med'], data['nzp'], 1)

    poly1d_fn = np.poly1d(coef)
    xarray = np.arange(-50, 50)
    ax2.plot(xarray, poly1d_fn(xarray), 'y--', zorder=10, label='slope = '+str(np.round(coef[0], 3)))
    ax2.plot(data['tzp_med'], data['nzp'], 'k.', alpha=0.1, zorder=0)
    rmstzp = np.sqrt(np.sum(data['tzp_med']**2)/len(data))
    rmsnzp = np.sqrt(np.sum(data['nzp']**2)/len(data))
    print('RMS TZP = ', rmstzp, ' m/s')
    print('RMS NZP = ', rmsnzp, ' m/s')
    pt2nzp, _ = spearmanr(data['tzp_med'], data['nzp'])
    print('TZP vs NZP Pearson corr. coeff = ', pt2nzp)
    ax2.set_xlabel(r'$TZP$ [m/s]')
    ax2.set_ylabel(r'$NZP$ [m/s]')
    #ax2.axis('equal')
    ax2.legend()
    ax2.set_ylim(-7, 7)
    ax2.set_xlim(-7, 7)
    ax2.set_aspect(1)

    #coef2 = np.polyfit(data['nzp'], dt, 1)
    coef2 = np.polyfit(dtout, nzpout,  1)
    #print(coef2)
    poly1d_fn2 = np.poly1d(coef2)
    xarray = np.arange(-50, 50)
    ax3.plot(xarray, poly1d_fn2(xarray), 'y--', label='slope = '+str(np.round(coef2[0], 3))+ r' m/s/mK', zorder=40)
    #ax3.plot(dt - np.mean(dt), data['nzp'], 'k.', alpha=0.1)
    ax3.plot(dtout, nzpout, 'k.', alpha=0.1)
    temp2nzp, _ = spearmanr(dtout, nzpout)
    print('Temp. grad. vs NZP Pearson corr. coeff = ', temp2nzp)
    ax3.set_ylabel(r'$NZP$ [m/s]')
    ax3.set_xlabel(r'$\nabla T$ [mK]')
    ax3.set_xlim(-0.01, 0.01)
    ax3.set_ylim(-7., 7.)
    ax3.legend()
    #ax3.set_ylim(0.495, 0.520)
    #ax3.axis('equal')
    plt.tight_layout()
    plt.savefig('plots/nzp_tzp_all_plot.png')
    plt.show()
    plt.clf()
    plt.close()


def dd_rv_per_obs_correct():
    out_path = 'data/stars/'  # _Aug17/'

    star_files = glob.glob(out_path + 'J*')
    quiet_stars = []
    rvscale = []
    nstars = 0
    nobs = 0
    tzp_all = pd.read_csv('data/all_stars_tzp.csv', sep=',')
    tzpmean = np.mean(tzp_all['tzp_med'].values)
    bjd, nzp, e_nzp, tzp, rvcout, avcout, star, sptype, e_nzp, e_tzp = [], [],[], [], [], [], [], [], [], []
    t1, t2, t3, t4, t5, t6, t7, t8 = [], [], [], [], [], [], [], []
    tzp1, tzp2, tzp3, tzp4 = [], [], [], []

    for i in range(len(star_files)):
        star_id = star_files[i][len(out_path):]
        star_id = star_id[:-11]
        print(star_id)
        star_data = pd.read_csv(star_files[i], sep=',')
        std_rvc = np.std(star_data['rvc'])
        # print(std_rvc)
        #star_data = star_data.loc[np.abs(star_data['tzp_median']) < 50]
        #print(star_data.columns)
        if std_rvc < 10:
            nstars += 1
            nobs += len(star_data)
            std_avc = np.std(star_data['avc'])
            #fdd = (star_data['dd_c_date'] - star_data['dd_moes_obs'])*star_data['pix2ms']
            #tvc = star_data['rvc'] + star_data['dd_m_date']*star_data['pix2ms'] + star_data['tzp']  #fdd
            tzp_test = star_data['tzp_median'] - tzpmean
            tvc = star_data['rvc'] + tzp_test
            std_tvc = np.std(tvc)
            quiet_stars.append(
                np.array([star_id, std_rvc, std_avc, std_tvc]))  # , std_tvc0, std_tvc1, std_tvc2, std_tvc3]))
            '''
            for k in range(len(star_data)):
                #fddaux = (star_data['dd_c_date'].values[k] - star_data['dd_moes_obs'].values[k]) * star_data['pix2ms'].values[k]
                e_tzp_aux = np.sqrt(star_data['e_tzp'].values[k] ** 2 + (star_data['dd_m_date'].values[k]*star_data['pix2ms'].values[k]) ** 2)
                nzp.append(star_data['nzp'].values[k])
                e_nzp.append(star_data['e_nzp'].values[k])
                bjd.append(star_data['bjd'].values[k])
                #tzp.append(star_data['tzp_med'].values)
                tzp.append(tzp_test[k])
                #e_tzp.append(star_data['e_tzp'].values[k])
                rvcout.append(star_data['rvc'].values[k])
                avcout.append(star_data['avc'].values[k])
                star.append(star_data['starid'].values[k])
                sptype.append(star_data['spt'].values[k])
                t1.append(star_data['t1'].values[k])
                t2.append(star_data['t2'].values[k])
                t3.append(star_data['t3'].values[k])
                t4.append(star_data['t4'].values[k])
                t5.append(star_data['t5'].values[k])
                t6.append(star_data['t6'].values[k])
                t7.append(star_data['t7'].values[k])
                t8.append(star_data['t8'].values[k])
            #tvc0 = star_data['rvc'] + star_data['tzp0']
            #std_tvc0 = np.std(tvc0)
            #tvc1 = star_data['rvc'] + star_data['tzp1']
            #std_tvc1 = np.std(tvc1)
            #tvc2 = star_data['rvc'] + star_data['tzp2']
            #std_tvc2 = np.std(tvc2)
            #tvc3 = star_data['rvc'] + star_data['tzp3']
            #std_tvc3 = np.std(tvc3)
            '''


    quiet_stars = np.array(quiet_stars)
    print(quiet_stars)
    #zps = zps.loc[zps.nzp > -4]
    #zps = zps.loc[zps.nzp < 4]
    print('Total number of stars = ', nstars)
    print('Total number of observations = ', nobs)
    #print('Total number of observing nights = ', ndates)

    #stars = np.unique(zps['star'].values)
    #quiet_stars = []
    #for i in range(len(stars)):
    #    zpdata = zps.loc[zps.star == stars[i]]
    #    tvc2 = zpdata['rvc'] - zpdata['tzp']
    #    std_tvc2 = np.std(tvc2)
    ##    std_avc2 = np.std(zpdata['avc'].values)
     #   std_rvc2 = np.std(zpdata['rvc'].values)
        #print(zpdata)
        #print(stars[i], std_rvc2, std_avc2, std_tvc2)
    #    quiet_stars.append(np.array([stars[i], std_rvc2, std_avc2, std_tvc2]))


    #quiet_stars = np.array(quiet_stars)
    #print(quiet_stars)
    '''
    for k in range(len(quiet_stars)):
        if quiet_stars[k][3] < quiet_stars[k][1]:
            ntvc += 1
            tvc_stars.append(quiet_stars[k])
        elif quiet_stars[k][3] < quiet_stars[k][2]:
            ntvc_plus += 1
            tvc_stars_plus.append(quiet_stars[k])
            #print(quiet_stars[k])

    print('Number of TVC well corrected stars = ', ntvc)
    print('Number of TVC super well corrected stars = ', ntvc_plus)
    print('Percentage of TVC well corrected stars = ', ntvc / nstars * 100, '%')
    print('Percentage of TVC super well corrected stars = ', ntvc_plus / nstars * 100, '%')
    '''
    #tvc_stars = np.array(tvc_stars)
    #tvc_stars_plus = np.array(tvc_stars_plus)

    #tvc_mean = np.mean(tvc_stars[:, 1].astype(np.float))
    #tvc2 = np.mean(tvc_stars[:, 3].astype(np.float))
    #print((tvc_mean - tvc2) * 100 / tvc_mean)
    #tvc_mean_plus = np.mean(tvc_stars_plus[:, 2].astype(np.float) - tvc_stars_plus[:, 3].astype(np.float))

    #tvc_stars_mean = np.mean(tvc_stars_plus[:, 1].astype(np.float))
    #tvc_stars_plus_mean = np.mean(tvc_stars_plus[:, 2].astype(np.float))
    #tvc_stars_plus2_mean = np.mean(tvc_stars_plus[:, 3].astype(np.float))

    #print(tvc_stars_mean, tvc_stars_plus_mean, tvc_stars_plus2_mean)
    print(quiet_stars)
    rvc_std = quiet_stars[:, 1].astype(np.float)
    avc_std = quiet_stars[:, 2].astype(np.float)
    tvc_std = quiet_stars[:, 3].astype(np.float)
    print(np.mean(rvc_std), np.mean(tvc_std), np.mean(avc_std))
    # PLOT

    fig, axes = plt.subplots(nrows=3,
                             ncols=1,
                             figsize=(14, 10),
                             gridspec_kw={'height_ratios': [1, 1, 1]},
                             sharex=True,
                             sharey=False)

    #plt.figure(figsize=[10, 3])
    binwidth = 0.5
    bins_all = np.arange(0., 10., binwidth)

    # RVC
    axes[0].hist(rvc_std, bins=bins_all, color='yellow', alpha=0.8, label='Non-corrected', edgecolor='black')
    axes[0].hist(avc_std, bins=bins_all, color='blue', alpha=0.5, label='NZP corrected', edgecolor='black')
    axes[0].legend(fontsize=16)
    hist_rvc, bins_rvc = np.histogram(rvc_std, bins=bins_all)
    yrange = [0, 10, 20, 30, 40]
    axes[0].set_yticklabels(yrange, fontsize=18)

    bins_rvc = bins_rvc[1:] - binwidth/2
    rvc_weighted_mean = np.sum(bins_rvc * (hist_rvc / len(rvc_std)))
    # AVC
    axes[1].hist(rvc_std, bins=bins_all, color='yellow', alpha=0.8, label='Non-corrected', edgecolor='black')
    axes[1].hist(tvc_std, bins=bins_all, color='red', alpha=0.5, label='TZP corrected', edgecolor='black')
    axes[1].set_ylabel('Number of stars', fontsize=20)
    axes[1].legend(fontsize=16)
    axes[1].set_yticklabels(yrange, fontsize=18)
    hist_avc, bins_avc = np.histogram(avc_std, bins=bins_all)
    bins_avc = bins_avc[1:] - binwidth/2
    avc_weighted_mean = np.sum(bins_avc * (hist_avc / len(avc_std)))

    # TVC

    axes[2].hist(avc_std, bins=bins_all, color='blue', alpha=0.5, label='NZP corrected', edgecolor='black')
    axes[2].hist(tvc_std, bins=bins_all, color='red', alpha=0.5, label='TZP corrected', edgecolor='black')
    axes[2].legend(fontsize=16)
    axes[2].set_yticklabels(yrange, fontsize=18)

    plt.xticks(fontsize=18)

    hist_tvc, bins_tvc = np.histogram(tvc_std, bins=bins_all)
    #print(bins_tvc)
    bins_tvc = bins_tvc[1:] - binwidth/2
    #print(weighted_average_m1(bins_tvc, hist_tvc))
    #print(weighted_average_m1(bins_rvc, hist_rvc))
    #print(weighted_average_m1(bins_avc, hist_avc))
    #weighted_average_m1(hist_tvc, )
    axes[2].plot(bins_tvc, hist_tvc, 'ro')
    tvc_weighted_mean = np.sum(bins_tvc * (hist_tvc / len(tvc_std)))

    # lineas verticales

    tzp_mean = [np.mean(tvc_std), np.mean(tvc_std)]
    nzp_mean = [np.mean(avc_std), np.mean(avc_std)]
    nozp_mean = [np.mean(rvc_std), np.mean(rvc_std)]
    yrange = [0., 100.]
    #axes[2].plot(tzp_mean, yrange, '--', 'red')
    #axes[2].plot(nzp_mean, yrange, '--', 'blue')


    #plt.hist(quiet_stars[:, 4].astype(np.float), bins=bins_all, color='green', alpha=0.5, label='TZP corrected', edgecolor='black')
    #plt.hist(quiet_stars[:, 5].astype(np.float), bins=bins_all, color='purple', alpha=0.5, label='TZP corrected', edgecolor='black')
    #plt.hist(quiet_stars[:, 3].astype(np.float), bins=bins_all, color='blue', alpha=0.5, label='TZP corrected', edgecolor='black')
    #plt.hist(quiet_stars[:, 7].astype(np.float), bins=bins_all, color='brown', alpha=0.5, label='TZP corrected', edgecolor='black')
    #hist_tvc, bins_tvc = np.histogram(quiet_stars[:, 3].astype(np.float), bins=bins_all)
    #bins_tvc = bins_tvc[1:]
    #tvc_weighted_mean = np.sum(bins_tvc * (hist_tvc / len(tvc_std)))

    # print(rvc_weighted_mean, avc_weighted_mean, tvc_weighted_mean)
    #plt.legend()
    plt.xlabel('RVs standard deviation [m/s]', fontsize=20)
    #plt.ylabel('Number of RV-quiet stars')
    # plt.savefig('plots/std_rvc_vs_avc.png')
    # plt.savefig('plots/std_avc_vs_tvc.png')
    # plt.savefig('plots/std_rvc_vs_tvc.png')
    # plt.savefig('plots/std_rvc_vs_avc_vs_tvc.png',bbox_inches='tight')
    plt.savefig('plots/std_rvc_vs_avc_vs_tvc_v1.png',
                bbox_inches='tight')
    plt.show()
    plt.clf()


def remove_outliers():
    #data = pd.read_csv('data/all_stars_tzp_filtered.csv', sep=',')
    data = pd.read_csv('data/all_stars_tzp_full.csv', sep=',')
    print(data)
    import numpy as np
    from scipy import stats

    # Generate a sample periodic dataset (replace this with your own data)
    # For demonstration purposes, we'll create a sine wave with outliers.
    np.random.seed(42)
    #x = np.linspace(0, 4 * np.pi, 100)
    #y = np.sin(x) + np.random.normal(0, 0.2, 100)
    x = data['bjd'].values
    y = data['nzp'].values

    # Define a threshold for identifying outliers (you can adjust this)
    z_score_threshold = 1.6

    # Calculate z-scores for the data points
    z_scores = np.abs(stats.zscore(y))

    # Identify outliers based on the z-score threshold
    outliers = np.where(z_scores > z_score_threshold)[0]

    # Remove the outliers from the dataset
    filtered_y = np.delete(y, outliers)
    filtered_x = np.delete(x, outliers)

    # Plot the original and filtered data for visualization
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'b.', label='Original Data')
    plt.scatter(x[outliers], y[outliers], color='red', marker='x', label='Outliers')
    plt.legend()
    plt.title('Original Data with Outliers')

    plt.subplot(2, 1, 2)
    plt.plot(filtered_x, filtered_y, 'g.', label='Filtered Data')
    plt.legend()
    plt.title('Filtered Data (Outliers Removed)')

    plt.tight_layout()
    plt.show()
    n = 0
    print(data.keys())
    bjd, nzp, tzpmed, tzpgaus, tzpweight, tzpmean = [], [], [], [], [], []
    rvc, avc = [], []
    stid = []
    t1, t2, t3, t4, t5, t6, t7, t8 = [], [], [], [], [], [], [], []
    for i in range(len(data)):
        for k in range(len(filtered_x)):
            if data['bjd'].values[i] == filtered_x[k]:
                #print('carlos')
                bjd.append(data['bjd'].values[i])
                nzp.append(data['nzp'].values[i])
                rvc.append(data['rvc'].values[i])
                avc.append(data['avc'].values[i])
                tzpmed.append(data['tzp_med'].values[i])
                tzpgaus.append(data['tzp_gauss'].values[i])
                tzpweight.append(data['tzp_weight'].values[i])
                tzpmean.append(data['tzp_mean'].values[i])
                t1.append(data['t1'].values[i])
                t2.append(data['t2'].values[i])
                t3.append(data['t3'].values[i])
                t4.append(data['t4'].values[i])
                t5.append(data['t5'].values[i])
                t6.append(data['t6'].values[i])
                t7.append(data['t7'].values[i])
                t8.append(data['t8'].values[i])
                stid.append(data['id'].values[i])
                n += 1


    all_data = pd.DataFrame()
    all_data['bjd'] = bjd
    all_data['nzp'] = nzp
    all_data['rvc'] = rvc
    all_data['avc'] = avc
    all_data['tzp_med'] = tzpmed
    all_data['tzp_gauss'] = tzpgaus
    all_data['tzp_weight'] = tzpweight
    all_data['tzp_mean'] = tzpmean
    all_data['t1'] = np.array(t1)
    all_data['t2'] = np.array(t2)
    all_data['t3'] = np.array(t3)
    all_data['t4'] = np.array(t4)
    all_data['t5'] = np.array(t5)
    all_data['t6'] = np.array(t6)
    all_data['t7'] = np.array(t7)
    all_data['t8'] = np.array(t8)
    all_data['id'] = np.array(stid)
    all_data.to_csv('all_stars_tzp_filtered.csv', index=False)


def temperature_plots():
    data = pd.read_csv('data/all_stars_tzp.csv', sep=',')

    fig, axes = plt.subplots(nrows=8,
                             ncols=1,
                             figsize=(12, 10),
                             #gridspec_kw={'height_ratios': [1, 0.5, 0.5]},
                             sharex=True,
                             sharey=False)

    avg_temp = (data['t1'] + data['t2'] + data['t3'] + data['t4'] + data['t5'] + data['t6'] + data['t7'] + data['t8'])/8.
    print(avg_temp)
    avg_temp_all = np.mean(avg_temp)
    line = [0, 1e8]
    yline = [avg_temp_all, avg_temp_all]
    axes[0].set_title(r'Temperature measurements')
    axes[0].plot(data['bjd'] - 2458000, data['t1'], 'k.', label=r'T$_1$')
    axes[0].plot(line, yline, 'r--', linewidth=3)
    axes[0].set_ylabel(r'T$_1$' + r'($^\circ C$)')
    axes[0].set_ylim(9.9, 11.1)
    axes[0].set_xlim(min(data['bjd'].values), max(data['bjd'].values))
    axes[1].plot(data['bjd'] - 2458000, data['t2'], 'k.', label=r'T$_2$')
    axes[1].plot(line, yline, 'r--', linewidth=3)
    axes[1].set_ylabel(r'T$_2$' + r'($^\circ C$)')
    axes[1].set_ylim(9.9, 11.1)
    axes[1].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[2].plot(data['bjd'] - 2458000, data['t3'], 'k.', label=r'T$_3$')
    axes[2].plot(line, yline, 'r--', linewidth=3)
    axes[2].set_ylabel(r'T$_3$' + r'($^\circ C$)')
    axes[2].set_ylim(9.9, 11.1)
    axes[2].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[3].plot(data['bjd'] - 2458000, data['t4'], 'k.', label=r'T$_4$')
    axes[3].plot(line, yline, 'r--', linewidth=3)
    axes[3].set_ylabel(r'T$_4$' + r'($^\circ C$)')
    axes[3].set_ylim(9.9, 11.1)
    axes[3].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[4].plot(data['bjd'] - 2458000, data['t5'], 'k.', label=r'T$_5$')
    axes[4].plot(line, yline, 'r--', linewidth=3)
    axes[4].set_ylabel(r'T$_5$' + r'($^\circ C$)')
    axes[4].set_ylim(9.9, 11.1)
    axes[4].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[5].plot(data['bjd'] - 2458000, data['t6'], 'k.', label=r'T$_6$')
    axes[5].plot(line, yline, 'r--', linewidth=3)
    axes[5].set_ylabel(r'T$_6$' + r'($^\circ C$)')
    axes[5].set_ylim(9.9, 11.1)
    axes[5].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[6].plot(data['bjd'] - 2458000, data['t7'], 'k.', label=r'TEMP$_7$')
    axes[6].plot(line, yline, 'r--', linewidth=3)
    axes[6].set_ylabel(r'T$_7$' + r'($^\circ C$)')
    axes[6].set_ylim(9.9, 11.1)
    axes[6].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[7].plot(data['bjd'] - 2458000, data['t8'], 'k.', label=r'TEMP$_8$')
    axes[7].plot(line, yline, 'r--', linewidth=3)
    axes[7].set_ylabel(r'T$_8$' + r'($^\circ C$)')
    axes[7].set_ylim(9.9, 11.1)
    axes[7].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[7].set_xlabel('BJD - 2458000 (days)')
    plt.savefig('plots/temperatures_ts.png')
    plt.show()


def delta_temperature_plots():
    data = pd.read_csv('data/all_stars_tzp.csv', sep=',')


    fig, axes = plt.subplots(nrows=6,
                             ncols=1,
                             figsize=(12, 10),
                             #gridspec_kw={'height_ratios': [1, 0.5, 0.5]},
                             sharex=True,
                             sharey=False)

    avg_temp = (data['t1'] + data['t2'] + data['t3'] + data['t4'] + data['t5'] + data['t6'] + data['t7'] + data['t8'])/8.
    print(avg_temp)
    data['temp_grad'] = (data['t1'] + data['t2'] + data['t3']) / 3. - (data['t6'] + data['t7'] + data['t8']) / 3
    avg_temp_all = np.mean(avg_temp)
    mean_temp_grad = np.mean(data['temp_grad'])
    line = [0, 1e8]
    deltay = 0.008
    yline = [mean_temp_grad, mean_temp_grad]
    rms0 = np.sqrt(np.sum((data['t4'] - data['t8']) ** 2) / len(data))
    std0 = np.std(data['t4'] - data['t8'])
    print(rms0)
    axes[0].set_title(r'Temperature differences')
    axes[0].plot(data['bjd'] - 2458000, data['t4'] - data['t8'] - np.mean(data['t4'] - data['t8']), 'k.', label=r'$\sigma$ = ' + str(float(np.round(std0,4))) + r'$^\circ$C')
    axes[0].set_ylabel(r'T$_4$ - T$_8$' + r'($^\circ C$)')
    axes[0].set_ylim(-deltay, deltay)
    axes[0].legend(loc='best')
    axes[0].set_xlim(min(data['bjd'].values), max(data['bjd'].values))

    #rms1 = np.sqrt(np.sum((data['t4'] - data['t5']) ** 2) / len(data))
    rms1 = np.std(data['t4'] - data['t5'])
    axes[1].plot(data['bjd'] - 2458000, data['t4'] - data['t5'] - np.mean(data['t4'] - data['t5']), 'k.', label=r'$\sigma$ = ' + str(float(np.round(rms1,4))) + r'$^\circ$C')
    axes[1].set_ylabel(r'T$_4$ - T$_5$' + r'($^\circ C$)')
    axes[1].set_ylim(-deltay, deltay)
    axes[1].legend(loc='best')

    axes[1].set_xlim(min(data['bjd'].values), max(data['bjd'].values))


    rms2 = np.sqrt(np.sum((data['t1'] - data['t2']) ** 2) / len(data))
    rms2 = np.std(data['t1'] - data['t2'])
    axes[2].plot(data['bjd'] - 2458000, data['t1'] - data['t2'] - np.mean(data['t1'] - data['t2']), 'k.',
                 label=r'$\sigma$ = ' + str(float(np.round(rms2,4))) + r'$^\circ$C')
    axes[2].set_ylabel(r'T$_1$ - T$_2$' + r'($^\circ C$)')
    axes[2].set_ylim(-deltay, deltay)
    axes[2].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[2].legend(loc='best')
    #axes[1].plot(data['bjd'] - 2458000, data['t7'] - data['t3'] - np.mean(data['t7'] - data['t3']), 'k.', label=r'T$_7$ - T$_3$')
    #axes[1].plot(line, yline, 'r--', linewidth=3)
    #axes[1].set_ylabel(r'T$_7$ - T$_3$')
    #axes[1].set_ylim(-deltay, deltay)
    #axes[1].set_ylim(9.9, 11.1) + r'($^\circ C$)'
    #axes[1].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    rms3 = np.sqrt(np.sum((data['t4'] - data['t1']) ** 2) / len(data))
    rms3 = np.std(data['t4'] - data['t1'])
    axes[3].plot(data['bjd'] - 2458000, data['t4'] - data['t1'] - np.mean(data['t4'] - data['t1']), 'k.',
                 label=r'$\sigma$ = ' + str(float(np.round(rms3, 4))) + r'$^\circ$C')
    axes[3].set_ylabel(r'T$_4$ - T$_1$' + r'($^\circ C$)')
    axes[3].set_ylim(-deltay, deltay)
    axes[3].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[3].legend(loc='best')
    #axes[3].set_ylim(9.9, 11.1)
    #axes[3].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    rms4 = np.sqrt(np.sum((data['t5'] - data['t2']) ** 2) / len(data))
    rms4 = np.std(data['t5'] - data['t2'])
    axes[4].plot(data['bjd'] - 2458000, data['t5'] - data['t2'] - np.mean(data['t5'] - data['t2']), 'k.', label=r'$\sigma$ = ' + str(float(np.round(rms4, 4))) + r'$^\circ$C')
    #axes[4].plot(line, yline, 'r--', linewidth=3)
    axes[4].set_ylabel(r'T$_5$ - T$_2$' + r'($^\circ C$)')
    axes[4].set_ylim(-deltay, deltay)
    #axes[4].set_ylim(9.9, 11.1)
    axes[4].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[4].legend(loc='best')
    #axes[4].set_xlabel('BJD - 2458000 (days)')

    rms5 = np.sqrt(np.sum((data['t7'] - data['t3']) ** 2) / len(data))
    rms5 = np.std(data['t7'] - data['t3'])
    axes[5].plot(data['bjd'] - 2458000, data['t7'] - data['t3'] - np.mean(data['t7'] - data['t3']), 'k.',
                 label=r'$\sigma$ = ' + str(float(np.round(rms5, 4))) + r'$^\circ$C')
    # axes[4].plot(line, yline, 'r--', linewidth=3)
    axes[5].set_ylabel(r'T$_7$ - T$_3$' + r'($^\circ C$)')
    # axes[4].set_ylim(9.9, 11.1)
    axes[5].set_ylim(-deltay, deltay)
    axes[5].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    axes[5].set_xlabel('BJD - 2458000 (days)')
    axes[5].legend(loc='best')
    #axes[5].plot(data['bjd'] - 2458000, data['t6'], 'k.', label=r'T$_6$')
    #axes[5].plot(line, yline, 'r--', linewidth=3)
    #axes[5].set_ylabel(r'T$_6$')
    ##axes[5].set_ylim(9.9, 11.1)
    #axes[5].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    #axes[6].plot(data['bjd'] - 2458000, data['t7'], 'k.', label=r'TEMP$_7$')
    #axes[6].plot(line, yline, 'r--', linewidth=3)
    #axes[6].set_ylabel(r'T$_7$')
    #axes[6].set_ylim(9.9, 11.1)
    #axes[6].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    #axes[6].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    #axes[7].plot(data['bjd'] - 2458000, data['t8'], 'k.', label=r'TEMP$_8$')
    #axes[7].plot(line, yline, 'r--', linewidth=3)
    #axes[7].set_ylabel(r'T$_8$')
    #axes[7].set_ylim(9.9, 11.1)
    #axes[7].set_xlim(min(data['bjd'].values) - 2458000, max(data['bjd'].values) - 2458000)
    plt.savefig('plots/temps_difference.png')
    plt.show()


def tzp_per_obs_correction():
    out_path = 'data/stars/'  # _Aug17/'

    star_files = glob.glob(out_path + 'J*')
    quiet_stars = []
    rvscale = []
    nstars = 0
    nobs = 0
    tzp_all = pd.read_csv('data/all_stars_tzp.csv', sep=',')
    tzpmean = np.mean(tzp_all['tzp_med'].values)
    bjd, nzp, e_nzp, tzp, rvcout, avcout, star, sptype, e_nzp, e_tzp = [], [],[], [], [], [], [], [], [], []
    t1, t2, t3, t4, t5, t6, t7, t8 = [], [], [], [], [], [], [], []
    tzp1, tzp2, tzp3, tzp4 = [], [], [], []

    for i in range(len(star_files)):
        star_id = star_files[i][len(out_path):]
        star_id = star_id[:-11]
        star_data = pd.read_csv(star_files[i], sep=',')
        if len(star_data) > 10:
            std_rvc = np.std(star_data['rvc'])
            if std_rvc < 10:
                nstars += 1
                nobs += len(star_data)
                #std_avc = np.std(star_data['avc'])
                #print(star_id)
                star.append(star_id)

        # print(std_rvc)
        #star_data = star_data.loc[np.abs(star_data['tzp_median']) < 50]
        #print(star_data.columns)

            #tvc = star_data['rvc'] + tzp_test
            #std_tvc = np.std(tvc)
            #quiet_stars.append(
            #    np.array([star_id, std_rvc, std_avc, std_tvc]))  # , std_tvc0, std_tvc1, std_tvc2, std_tvc3]))
    print(nstars)

    for s in star:
        starpath = 'data/stars/'
        stardata = pd.read_csv(starpath + s + '_avc_dd.csv', sep=',')
        tzpmean = np.mean(stardata['tzp_median'])
        tzp = stardata['tzp_median'] - tzpmean
        tvc = stardata['rvc'] + tzp
        std_rvc = np.std(stardata['rvc'])
        std_tvc = np.std(tvc)
        std_avc = np.std(stardata['avc'])
        quiet_stars.append(np.array([s, std_rvc, std_avc, std_tvc]))

    ntvc = 0.
    ntvc_plus = 0.

    tvc_stars = []
    tvc_stars_plus = []
    for k in range(len(quiet_stars)):
        if quiet_stars[k][3] < quiet_stars[k][1]:
            ntvc += 1
            tvc_stars.append(quiet_stars[k])
        elif quiet_stars[k][3] < quiet_stars[k][2]:
            ntvc_plus += 1
            tvc_stars_plus.append(quiet_stars[k])
            # print(quiet_stars[k])

    print('Number of stars = ', len(quiet_stars))
    print('Number of TVC well corrected stars = ', ntvc)
    print('Number of TVC super well corrected stars = ', ntvc_plus)
    print('Percentage of TVC well corrected stars = ', ntvc / nstars * 100, '%')
    print('Percentage of TVC super well corrected stars = ', ntvc_plus / nstars * 100, '%')


if __name__ == '__main__':
    date = '2017-10-20'
    fib = 'A'
    import optimization
    #optimization.full_fit_date(date, 'hcl', fib)
    full_model_residuals(date, fib)
    #echellogram_plot(date, fib)
    #make_dd_hcl_file()

    #drifts_plot()
    #ddplots()

    #rvs_files_dd_per_observation_final_v2()

    #remove_outliers()
    #tzp_vs_nzp_plot()

    #dd_rv_per_obs_correct()
    #tzp_per_obs_correction()

    #temps_plots()
    #temperature_plots()
    #delta_temperature_plots()