import matplotlib as mpl
# mpl.use('qt4agg')
from optics import env_data
import matplotlib.pyplot as plt
import pymultinest
import numpy as np
import utils
from optics import parameters
from optics import vis_spectrometer
import time
import ws_load
import os
import pandas as pd
import dynesty
import corner
from astropy.time import Time
import do_cornerplot


def rewrite_best_fit_params(date):
    pars_index = [42, 11, 12, 9, 13, 10, 35, 17, 28, 21, 33, 24, 29, 36, 39]
    bestfit_file = pd.read_csv('plots/posteriors/'+str(date)+'/best_fit_params.tsv', names=['pars'])
    print(bestfit_file)
    init_state_a = parameters.load_sa(date, 'a')
    print(init_state_a)
    for k in range(len(pars_index)):
        init_state_a[pars_index[k]] = bestfit_file['pars'].values[k]

    parameters.write_sim(date, init_state_a, 'a')


# We load our data set
def carmenes_vis(date, fiber):
    par_ini = parameters.load_date(fiber, date)
    # write_old = parameters.write_old(par_ini)

    # We load the data
    wsa_data, wsb_data = ws_load.carmenes_vis_ws_for_fit(date)
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)

    if fiber == 'a':
        spec = ws_load.spectrum_from_ws(wsa_data)
        fib = 'A'
    else:
        spec = ws_load.spectrum_from_ws(wsb_data)
        fib = 'B'

    temps = env_data.get_temps_date(date)

    init_state = parameters.load_sa(str(fiber))
    pressure = env_data.get_p_date(date)
    init_state[-1] = pressure
    print(init_state)
    if fiber == 'a':
        ws_model = vis_spectrometer.tracing(spec, init_state, 'A', temps)
        # ws_data, ws_model = sigma_clip(wsa_data, wsa_model, 5)
        ws_data = wsa_data
    else:
        ws_model = vis_spectrometer.tracing(spec, init_state, 'B', temps)
        ws_data = wsb_data
        # ws_data, ws_model = sigma_clip(wsb_data, wsb_model, 5)

    plt.plot(ws_data[:, 3], ws_data[:, 5], 'k+')
    plt.plot(ws_model[:, 2], ws_model[:, 3], 'r+')
    # plt.show()

    y = ws_data
    x = spec
    sigma_fit_x = ws_data[:, 4]

    plt.plot(ws_data[:, 3], ws_data[:, 5] - ws_model[:, 3], 'k.')
    # plt.show()
    plt.clf()

    plt.plot(ws_data[:, 3], ws_data[:, 3] - ws_model[:, 2], 'k.')
    # plt.show()
    plt.clf()

    # Define the prior (you have to transform your parameters, that come from the unit cube,
    # to the prior you want):

    def prior(cube, ndim, nparams):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 1e-4
        delta1 = 1e-6
        delta2 = 1e-5
        cube[0] = utils.transform_uniform(cube[0], par_ini[42] - delta1, par_ini[42] + delta1)  # ccd tilt z
        cube[1] = utils.transform_uniform(cube[1], par_ini[11] - delta0, par_ini[11] + delta0)  # echelle G
        cube[2] = utils.transform_uniform(cube[2], par_ini[12] - delta0, par_ini[12] + delta0)  # echelle blaze
        cube[3] = utils.transform_uniform(cube[3], par_ini[9] - delta0, par_ini[9] + delta0)  # coll tilt x
        cube[4] = utils.transform_uniform(cube[4], par_ini[13] - delta0, par_ini[13] + delta0)  # echelle gamma
        cube[5] = utils.transform_uniform(cube[5], par_ini[10] - delta0, par_ini[10] + delta0)  # coll tilt y
        cube[6] = utils.transform_uniform(cube[6], par_ini[35] - delta0, par_ini[35] + delta0)  # ccd ff tilt z
        cube[7] = utils.transform_uniform(cube[7], par_ini[17] - delta0, par_ini[17] + delta0)  # trf mirror tilt y
        cube[8] = utils.transform_uniform(cube[8], par_ini[28] - delta0, par_ini[28] + delta0)  # cam tilt x
        cube[9] = utils.transform_uniform(cube[9], par_ini[21] - delta0, par_ini[21] + delta0)  # grm tilt x
        cube[10] = utils.transform_uniform(cube[10], par_ini[33] - delta0, par_ini[33] + delta0)  # ccd ff tilt x
        cube[11] = utils.transform_uniform(cube[11], par_ini[24] - delta0, par_ini[24] + delta0)  # grm apex
        cube[12] = utils.transform_uniform(cube[12], par_ini[29] - delta0, par_ini[29] + delta0)  # cam tilt y
        cube[13] = utils.transform_uniform(cube[13], par_ini[36] - delta0, par_ini[36] + delta0)  # d ff ccd
        cube[14] = utils.transform_uniform(cube[14], par_ini[39] - delta0, par_ini[39] + delta0)  # ccd defocus

    # Define the likelihood:

    def loglike(cube, ndim, nparams):
        # Load parameters
        pars = parameters.load_sa(fiber)
        pars[42] = cube[0]  # ccd tilt z
        pars[11] = cube[1]  # echelle G
        pars[12] = cube[2]  # echelle blaze
        pars[9] = cube[3]  # coll tilt x
        pars[13] = cube[4]  # echelle gamma
        pars[10] = cube[5]  # coll tilt y
        pars[35] = cube[6]  # ccd_ff_tilt_z
        pars[17] = cube[7]  # trf mirror tilt y
        pars[28] = cube[8]  # cam tilt x
        pars[21] = cube[9]  # grm tilt x
        pars[33] = cube[10]  # ccd ff tilt x
        pars[24] = cube[11]  # grm apex
        pars[29] = cube[12]  # cam tilt y
        pars[36] = cube[13]  # d ff ccd
        pars[39] = cube[14]  # ccd defocus

        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)

        # Evaluate the log-likelihood:
        sigma_fit_y = np.full(len(y), .01)
        sigma_fit_x = np.full(len(y), .01)

        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + (
                -0.5 * ((model[:, 2] - y[:, 3]) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 15
    #path = '/home/eduspec/Documentos/moes/v3.1/vis/ns_moes/'

    #if not os.path.exists(path):
    #    os.makedirs(path)
    out_file = '/luthien/carmenes/vis/params/posteriors/' + str(date) + '/carm_vis_' + fiber 

    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points=300, outputfiles_basename=out_file, resume=False,
                    verbose=False)

    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    print(bestfit_params['parameters'])
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    outdir = '/luthien/carmenes/vis/params/posteriors/' + str(date) + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    mc_samples_data = pd.DataFrame(mc_samples)
    mc_samples_data.to_csv(outdir + 'samples.tsv', sep=',', index=False)
    #samples_out = pd.read_csv(outdir + 'samples.tsv')
    '''
    # print('Multicarlos optimization duration : %.3f hr' % (float(t2)))
    import corner

    posterior_names = []
    pars_index = [42, 11, 12, 9, 13, 10, 35, 17, 28, 21, 33, 24, 29, 36, 39]
    new_pars = init_state
    i = 0
    for par in pars_index:
        posterior_names.append(parameters.get_name(par))
        new_pars[par] = bestfit_params['parameters'][i]
        print(parameters.get_name(par), init_state[par], new_pars[par], bestfit_params['parameters'][i])
        i += 1

    first_time = True
    # for i in range(n_params):
    #    if first_time:
    #        posterior_data = output.posteriors['posterior_samples'][i]
    #        first_time = False
    #    else:
    #        posterior_data  = np.vstack((posterior_data, results.posteriors['posterior_samples'][i]))

    # posterior_data = posterior_data.T
    samples_names = [r'CCD T$_z$ (deg)', r'G$_{ech}$(mm$^{-1}$)', r'$\theta_{Blaze}$(m/s)',
                     r'Coll T$_x$ (deg)', r'$\gamma_{ech}$~(deg)', r'Coll T$_y$ (deg)',
                     r'CCD-FF T$_z$ (deg)', r'TM T$_{y}$ (deg)', r'Cam. T$_{x}$(deg)',
                     r'Grism T$_{x}$(deg)', r'CCD-FF T$_{x}$(deg)', r'Grism Apex~[(deg)',
                     r'Cam. T$_{y}$', r'd$_{FF-CCD}$~(mm)', r'CCD defocus~(mm)']

    #samples = pd.read_csv('plots/posteriors/2017-10-20/mc_samples.tsv', sep=',', names=samples_names)
    samples = pd.DataFrame(mc_samples, columns=samples_names)
    best_ns_pars = pd.read_csv('plots/posteriors/2017-10-20/best_fit_params.tsv', names=['pars'])
    samples_plot = pd.DataFrame()
    samples_plot[r'CCD T$_z$ (deg)'] = samples[r'CCD T$_z$ (deg)']
    samples_plot[r'G$_{ech}$(mm$^{-1}$)'] = samples[r'G$_{ech}$(mm$^{-1}$)']
    samples_plot[r'$\theta_{Blaze}$(m/s)'] = samples[r'$\theta_{Blaze}$(m/s)']
    samples_plot[r'Coll T$_x$ (deg)'] = samples[r'Coll T$_x$ (deg)']
    samples_plot[r'$\gamma_{ech}$~(deg)'] = samples[r'$\gamma_{ech}$~(deg)']
    samples_plot[r'Coll T$_y$ (deg)'] = samples[r'Coll T$_y$ (deg)']
    samples_plot[r'CCD-FF T$_z$ (deg)'] = samples[r'CCD-FF T$_z$ (deg)']
    samples_plot[r'TM T$_{y}$ (deg)'] = samples[r'TM T$_{y}$ (deg)']
    samples_plot[r'Cam. T$_{x}$(deg)'] = samples[r'Cam. T$_{x}$(deg)']
    samples_names_plot = [r'CCD T$_z$ (deg)', r'G$_{ech}$(mm$^{-1}$)', r'$\theta_{Blaze}$(m/s)',
                          r'Coll T$_x$ (deg)', r'$\gamma_{ech}$~(deg)', r'Coll T$_y$ (deg)',
                          r'CCD-FF T$_z$ (deg)', r'TM T$_{y}$ (deg)', r'Cam. T$_{x}$(deg)']

    pars_index = [42, 11, 12, 9, 13, 10, 35, 17, 28]  # , 21, 33, 24, 29, 36, 39]
    date = '2017-10-20'
    pars = parameters.load_sa(date, 'a')
    best_fit_sa = []
    for k in range(len(pars_index)):
        best_fit_sa.append(pars[pars_index[k]])

    # best_fit_par_dyn = [-2.08231, 13.37869, 40.68816416, 456.53436023, 0.03549355, 70.53001648, 153.82247384, 13.90580052, 2289.06760189, 0.27982054, 253.14207864, 11.37663006]
    # best_fit_par_stab = [-2.24, 13.22, 40.09, 456.81, 0.05, 89.68, 134.72, 12.59, 2369.76, 0.22, 210.58, 62.22]
    quantiles = corner.quantile(samples_plot, 0.6827)
    print(quantiles)
    figure = corner.corner(samples_plot,
                           labels=samples_names_plot,
                           bins=25,
                           color='k',
                           reverse=False,
                           levels=(0.6827, 0.9545, 0.9973),
                           plot_contours=True,
                           show_titles=True,
                           title_fmt=".2E",
                           title_kwargs={"fontsize": 10},
                           truths=best_fit_sa,
                           dpi=400,
                           truth_color='r',
                           scale_hist=True,
                           no_fill_contours=True,
                           plot_datapoints=True)
    '''
    #k = 0

    #corner.overplot_points(figure, best_ns_pars['pars'].values.T[None], marker="s", color="b")
    # for k in range(len(best_ns_pars)):
    #    corner.overplot_points(figure, best_ns_pars['pars'].values[k][None], marker="s", color="b")

    #plt.tight_layout()
    #plt.savefig('plots/posteriors/2017-10-20/corner_final.png')
    # plt.show()
    #figure = corner.corner(mc_samples, labels=posterior_names)
    #plt.savefig(outdir + 'moes_ins_corner_' + fiber + '_' + date + '.png')


def carmenes_vis_v2(date, fiber):
    wsa_data, wsb_data = ws_load.carmenes_vis_ws_for_fit(date)
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    pressure = env_data.get_p_date(date)
    init_state_a = parameters.load_sa(date, 'a')
    init_state_b = parameters.load_sa(date, 'b')
    init_state_a[-1] = pressure
    init_state_b[-1] = pressure
    temps = env_data.get_temps_date(date)
    wsa_model = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    wsb_model = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)
    res_x_a = np.sqrt(np.mean((wsa_data[:, 3] - wsa_model[:, 2]) ** 2))
    res_y_a = np.sqrt(np.mean((wsa_data[:, 5] - wsa_model[:, 3]) ** 2))
    res_x_b = np.sqrt(np.mean((wsb_data[:, 3] - wsb_model[:, 2]) ** 2))
    res_y_b = np.sqrt(np.mean((wsb_data[:, 5] - wsb_model[:, 3]) ** 2))

    print('Initial residuals')
    print('Fiber A')
    print('res_x =', res_x_a, ', res_y = ', res_y_a)
    print('Fiber B')
    print('res_x =', res_x_b, ', res_y = ', res_y_b)

    # We do only fiber A
    if fiber == 'a':
       y = wsa_data
       x = spec_a
       par_ini = init_state_a.copy()
    elif fiber == 'b':
       y = wsb_data
       x = spec_b
       par_ini = init_state_b.copy()


    def prior(cube, ndim, nparams):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[42], 7.5e-6)  # ccd tilt z
        cube[1] = utils.transform_normal(cube[1], par_ini[11], delta0)  # echelle G
        cube[2] = utils.transform_normal(cube[2], par_ini[12], 5.e-5)  # echelle blaze
        cube[3] = utils.transform_normal(cube[3], par_ini[9], delta2)  # coll tilt x
        cube[4] = utils.transform_normal(cube[4], par_ini[13], delta2)  # echelle gamma
        cube[5] = utils.transform_normal(cube[5], par_ini[10], delta2)  # coll tilt y
        cube[6] = utils.transform_normal(cube[6], par_ini[35], 5.e-3)  # ccd ff tilt z
        cube[7] = utils.transform_normal(cube[7], par_ini[17], 2*delta2)  # trf mirror tilt y
        cube[8] = utils.transform_normal(cube[8], par_ini[28], delta0)  # cam tilt x
        # cube[9] = utils.transform_normal(cube[9], par_ini[21], delta0)  # grm tilt x
        # cube[10] = utils.transform_normal(cube[10], par_ini[33], delta0)  # ccd ff tilt x
        # cube[11] = utils.transform_normal(cube[11], par_ini[24], delta2)  # grm apex
        # cube[12] = utils.transform_normal(cube[12], par_ini[29], delta2)  # cam tilt y
        # cube[13] = utils.transform_normal(cube[13], par_ini[36], delta2)  # d ff ccd
        # cube[14] = utils.transform_normal(cube[14], par_ini[39], delta2)  # ccd defocus

    def loglike(cube, ndim, nparams):
        # Load parameters
        if fiber == 'a':
           pars = parameters.load_sa(date, 'a')
        elif fiber == 'b':
           pars = parameters.load_sa(date, 'b')

        pars[42] = cube[0]  # ccd tilt z
        pars[11] = cube[1]  # echelle G
        pars[12] = cube[2]  # echelle blaze
        pars[9] = cube[3]  # coll tilt x
        pars[13] = cube[4]  # echelle gamma
        pars[10] = cube[5]  # coll tilt y
        pars[35] = cube[6]  # ccd_ff_tilt_z
        pars[17] = cube[7]  # trf mirror tilt y
        pars[28] = cube[8]  # cam tilt x
        #pars[21] = cube[9]  # grm tilt x
        #pars[33] = cube[10]  # ccd ff tilt x
        #pars[24] = cube[11]  # grm apex
        #pars[29] = cube[12]  # cam tilt y
        #pars[36] = cube[13]  # d ff ccd
        #pars[39] = cube[14]  # ccd defocus

        if len(pars) < 43:
            print('chafa')
        # Generate model:
        if fiber == 'a':
           model = vis_spectrometer.tracing(x, pars, 'A', temps)
        elif fiber == 'b':
           model = vis_spectrometer.tracing(x, pars, 'B', temps)

        # Evaluate the log-likelihood:
        #sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = y[:, 4]
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model[:, 2] - y[:, 3]) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model[:, 3] - y[:, 5]) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 9
    path = "".join(['/luthien/carmenes/vis/ns_results/', date, '/'])

    if not os.path.exists(path):
        os.makedirs(path)
    out_file = "".join([path, 'ns_fit_'+str(fiber)+'_'])

    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points=300, outputfiles_basename=out_file, resume=False,
                    verbose=False)

    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    #outdir = path
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    bestfitout = pd.DataFrame(bestfit_params['parameters'])
    bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    samplesout = pd.DataFrame(mc_samples)
    samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    #figure = corner.corner(mc_samples)  # , labels=posterior_names)
    #plt.tight_layout()
    #plt.savefig(outdir + 'parameters_cornerplot_'+str(fiber)+'.png')
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    print('Nested sampling of instrumental parameters for date ', date, ' done.')


def carmenes_vis_multinest(date, fiber):
    if fiber == 'a':
        fib = 'A'
    elif fiber == 'b':
        fib = 'B'
    wsa_data, wsb_data = ws_load.load_ws_for_fit(date)
    #wsa_data = np.array(wsa_data)
    #wsb_data = np.array(wsb_data)

    pressure = env_data.get_P_at_ws(date)
    init_state = parameters.load_date(fib, date)
    init_state[-1] = pressure
    temps = env_data.get_T_at_ws(date)
    if fiber == 'a':
        ws_data = wsa_data

        spec = ws_load.spectrum_from_data(wsa_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'A', temps)
        print(len(ws_model))

        fib = 'A'

    elif fiber == 'b':
        ws_data = wsb_data
        spec = ws_load.spectrum_from_data(wsb_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'B', temps)
        fib = 'B'

    res_x = np.sqrt(np.sum((ws_data['posm'].values - ws_model['x'].values)**2) / len(ws_data))
    res_y = np.sqrt(np.sum((ws_data['posmy'].values - ws_model['y'].values) ** 2) / len(ws_data))

    print('Initial residuals, fiber = ', fiber)
    print('res_x =', res_x, ', res_y = ', res_y)
    print(len(ws_data), len(ws_model))
    plt.plot(ws_data['posm'], ws_data['posm'].values - ws_model['x'].values, 'k.', zorder=10)
    plt.plot(ws_data['posm'], ws_data['posm'].values - ws_data['posc'].values, 'b.', zorder=0, alpha=0.5)
    plt.show()
    # We do only fiber A
    y = ws_data
    x = spec
    par_ini = init_state.copy()


    def prior(cube, ndim, nparams):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[11], delta0)  # ccd tilt z
        cube[1] = utils.transform_normal(cube[1], par_ini[9], delta0)  # echelle G
        cube[2] = utils.transform_normal(cube[2], par_ini[12], delta0)  # coll tilt x
        cube[3] = utils.transform_normal(cube[3], par_ini[21], delta0)  # echelle blaze
        cube[4] = utils.transform_normal(cube[4], par_ini[28], delta0)  # grism tilt x
        cube[5] = utils.transform_normal(cube[5], par_ini[10], delta0)  # camera x tilt
        cube[6] = utils.transform_normal(cube[6], par_ini[13], delta0)  # collimator y-tilt
        cube[7] = utils.transform_normal(cube[7], par_ini[1], delta0)  # echelle gamma angle
        cube[8] = utils.transform_normal(cube[8], par_ini[32], delta0)  # cam tilt x
        # cube[9] = utils.transform_normal(cube[9], par_ini[21], delta0)  # grm tilt x
        # cube[10] = utils.transform_normal(cube[10], par_ini[33], delta0)  # ccd ff tilt x
        # cube[11] = utils.transform_normal(cube[11], par_ini[24], delta2)  # grm apex
        # cube[12] = utils.transform_normal(cube[12], par_ini[29], delta2)  # cam tilt y
        # cube[13] = utils.transform_normal(cube[13], par_ini[36], delta2)  # d ff ccd
        # cube[14] = utils.transform_normal(cube[14], par_ini[39], delta2)  # ccd defocus

        return cube

    def loglike(cube, ndim, nparams):
        # Load parameters

        pars = parameters.load_date(fib, date)
        # print(pars[0])
        # print(cube)
        pars[11] = cube[0]
        pars[9] = cube[1]
        pars[12] = cube[2]
        pars[21] = cube[3]
        pars[28] = cube[4]
        pars[10] = cube[5]
        pars[13] = cube[6]
        pars[1] = cube[7]
        pars[32] = cube[8]
        # pars[21] = cube[9]  # grm tilt x
        # pars[33] = cube[10]  # ccd ff tilt x
        # pars[24] = cube[11]  # grm apex
        # pars[29] = cube[12]  # cam tilt y
        # pars[36] = cube[13]  # d ff ccd
        # pars[39] = cube[14]  # ccd defocus

        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)
        # Evaluate the log-likelihood:
        # sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = np.full(len(model), 0.1)
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model['x'].values - y['posm'].values) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model['y'].values - y['posmy'].values) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 9
    path = "".join(['data/posteriors/' + date + '/'])
    if not os.path.exists(path):
        os.makedirs(path)
    out_file = "".join([path, 'dfit_' + str(fiber) + ''])

    # Run MultiNest:

    pymultinest.run(loglike, prior, n_params, n_live_points=500, outputfiles_basename=out_file, resume=True,
                    verbose=True)


    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    maxpars = output.get_mode_stats()['modes'][0]['maximum']
    print(maxpars[0])
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    samples_names = [r'Ech. G [mm$^{-1}$]', r'Coll x-tilt [deg]', r'Ech. inc. angle [deg]',
                     r'Grism x-tilt [deg]', r'Cam x-tilt [deg]', r'Coll y-tilt [deg]',
                     r'Ech. $\gamma$-angle [deg]', r'Slit x-dec [mm]', r'Field flattener y-dec [mm]']


    mc_samples_data = pd.DataFrame(mc_samples)
    mc_samples_data.to_csv(path + 'mc_samples.tsv', sep=',', index=False, header=samples_names)

    bestfitout = pd.DataFrame(bestfit_params['parameters'])
    bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    pars = parameters.load_date(fib, date)

    pars[11] = bestfit_params['parameters'][0]
    pars[9] = bestfit_params['parameters'][1]
    pars[12] = bestfit_params['parameters'][2]
    pars[21] = bestfit_params['parameters'][3]
    pars[28] = bestfit_params['parameters'][4]
    pars[10] = bestfit_params['parameters'][5]
    pars[13] = bestfit_params['parameters'][6]
    pars[1] = bestfit_params['parameters'][7]
    pars[32] = bestfit_params['parameters'][8]
    #pars[42] = maxpars[0]
    #pars[11] = maxpars[1]
    #pars[9] = maxpars[2]
    #pars[12] = maxpars[3]
    #pars[21] = maxpars[4]
    #pars[28] = maxpars[5]
    #pars[10] = maxpars[6]
    #pars[13] = maxpars[7]
    #pars[1] = maxpars[8]
    #parameters.write_sa(pars, date, fiber)
    #fit_sa.moes_carmenes_vis_old(date)
    do_cornerplot.do_plot(date)
    # figure = corner.corner(mc_samples)  # , labels=posterior_names)
    # Get output:
    #output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    #bestfit_params = output.get_best_fit()
    #mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    #outdir = path
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    #bestfitout = pd.DataFrame(bestfit_params['parameters'])
    #bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    #figure = corner.corner(mc_samples)  # , labels=posterior_names)
    #plt.tight_layout()
    #plt.savefig(outdir + 'parameters_cornerplot_'+str(fiber)+'.png')
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    print('Multinest of instrumental parameters for date ', date, ' done.')


def carmenes_vis_multinest_full(date, fiber):
    if fiber == 'a':
        fib = 'A'
    elif fiber == 'b':
        fib = 'B'
    wsa_data, wsb_data = ws_load.load_ws_for_fit(date)
    #wsa_data = np.array(wsa_data)
    #wsb_data = np.array(wsb_data)

    pressure = env_data.get_P_at_ws(date)
    init_state = parameters.load_date(fib, date)
    init_state[-1] = pressure
    temps = env_data.get_T_at_ws(date)
    if fiber == 'a':
        ws_data = wsa_data

        spec = ws_load.spectrum_from_data(wsa_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'A', temps)
        print(len(ws_model))

        fib = 'A'

    elif fiber == 'b':
        ws_data = wsb_data
        spec = ws_load.spectrum_from_data(wsb_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'B', temps)
        fib = 'B'

    res_x = np.sqrt(np.sum((ws_data['posm'].values - ws_model['x'].values)**2) / len(ws_data))
    res_y = np.sqrt(np.sum((ws_data['posmy'].values - ws_model['y'].values) ** 2) / len(ws_data))

    print('Initial residuals, fiber = ', fiber)
    print('res_x =', res_x, ', res_y = ', res_y)
    print(len(ws_data), len(ws_model))
    plt.plot(ws_data['posm'], ws_data['posm'].values - ws_model['x'].values, 'k.', zorder=10)
    plt.plot(ws_data['posm'], ws_data['posm'].values - ws_data['posc'].values, 'b.', zorder=0, alpha=0.5)
    plt.show()
    # We do only fiber A
    y = ws_data
    x = spec
    par_ini = init_state.copy()


    def prior(cube, ndim, nparams):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[0], delta0)
        cube[1] = utils.transform_normal(cube[1], par_ini[1], delta0)
        cube[2] = utils.transform_normal(cube[2], par_ini[2], delta0)
        cube[3] = utils.transform_normal(cube[3], par_ini[3], delta0)
        cube[4] = utils.transform_normal(cube[4], par_ini[4], delta0)
        cube[5] = utils.transform_normal(cube[5], par_ini[5], delta0)
        cube[6] = utils.transform_normal(cube[6], par_ini[6], delta0)
        cube[7] = utils.transform_normal(cube[7], par_ini[7], delta0)
        cube[8] = utils.transform_normal(cube[8], par_ini[8], delta0)
        cube[9] = utils.transform_normal(cube[9], par_ini[9], delta0)
        cube[10] = utils.transform_normal(cube[10], par_ini[10], delta0)
        cube[11] = utils.transform_normal(cube[11], par_ini[11], delta0)
        cube[12] = utils.transform_normal(cube[12], par_ini[12], delta0)
        cube[13] = utils.transform_normal(cube[13], par_ini[13], delta0)
        cube[14] = utils.transform_normal(cube[14], par_ini[14], delta0)
        cube[15] = utils.transform_normal(cube[15], par_ini[15], delta0)
        cube[16] = utils.transform_normal(cube[16], par_ini[16], delta0)
        cube[17] = utils.transform_normal(cube[17], par_ini[17], delta0)
        cube[18] = utils.transform_normal(cube[18], par_ini[18], delta0)
        cube[19] = utils.transform_normal(cube[19], par_ini[19], delta0)
        cube[20] = utils.transform_normal(cube[20], par_ini[20], delta0)
        cube[21] = utils.transform_normal(cube[21], par_ini[21], delta0)
        cube[22] = utils.transform_normal(cube[22], par_ini[22], delta0)
        cube[23] = utils.transform_normal(cube[23], par_ini[23], delta0)
        cube[24] = utils.transform_normal(cube[24], par_ini[24], delta0)
        cube[25] = utils.transform_normal(cube[25], par_ini[25], delta0)
        cube[26] = utils.transform_normal(cube[26], par_ini[26], delta0)
        cube[27] = utils.transform_normal(cube[27], par_ini[27], delta0)
        cube[28] = utils.transform_normal(cube[28], par_ini[28], delta0)
        cube[29] = utils.transform_normal(cube[29], par_ini[29], delta0)
        cube[30] = utils.transform_normal(cube[30], par_ini[30], delta0)
        cube[31] = utils.transform_normal(cube[31], par_ini[31], delta0)
        cube[32] = utils.transform_normal(cube[32], par_ini[32], delta0)
        cube[33] = utils.transform_normal(cube[33], par_ini[33], delta0)
        cube[34] = utils.transform_normal(cube[34], par_ini[34], delta0)
        cube[35] = utils.transform_normal(cube[35], par_ini[35], delta0)
        cube[36] = utils.transform_normal(cube[36], par_ini[36], delta0)
        cube[37] = utils.transform_normal(cube[37], par_ini[37], delta0)
        cube[38] = utils.transform_normal(cube[38], par_ini[38], delta0)
        cube[39] = utils.transform_normal(cube[39], par_ini[39], delta0)
        cube[40] = utils.transform_normal(cube[40], par_ini[40], delta0)
        cube[41] = utils.transform_normal(cube[41], par_ini[41], delta0)
        cube[42] = utils.transform_normal(cube[42], par_ini[42], delta0)
        cube[43] = utils.transform_normal(cube[43], par_ini[43], delta0)
        return cube

    def loglike(cube, ndim, nparams):
        # Load parameters

        pars = parameters.load_date(fib, date)
        # print(pars[0])
        # print(cube)
        pars[0] = cube[0]
        pars[1] = cube[1]
        pars[2] = cube[2]
        pars[3] = cube[3]
        pars[4] = cube[4]
        pars[5] = cube[5]
        pars[6] = cube[6]
        pars[7] = cube[7]
        pars[8] = cube[8]
        pars[9] = cube[9]
        pars[10] = cube[10]
        pars[11] = cube[11]
        pars[12] = cube[12]
        pars[13] = cube[13]
        pars[14] = cube[14]
        pars[15] = cube[15]
        pars[16] = cube[16]
        pars[17] = cube[17]
        pars[18] = cube[18]
        pars[19] = cube[19]
        pars[20] = cube[20]
        pars[21] = cube[21]
        pars[22] = cube[22]
        pars[23] = cube[23]
        pars[24] = cube[24]
        pars[25] = cube[25]
        pars[26] = cube[26]
        pars[27] = cube[27]
        pars[28] = cube[28]
        pars[29] = cube[29]
        pars[30] = cube[30]
        pars[31] = cube[31]
        pars[32] = cube[32]
        pars[33] = cube[33]
        pars[34] = cube[34]
        pars[35] = cube[35]
        pars[36] = cube[36]
        pars[37] = cube[37]
        pars[38] = cube[38]
        pars[39] = cube[39]
        pars[40] = cube[40]
        pars[41] = cube[41]
        pars[42] = cube[42]
        pars[43] = cube[43]

        # pars[21] = cube[9]  # grm tilt x
        # pars[33] = cube[10]  # ccd ff tilt x
        # pars[24] = cube[11]  # grm apex
        # pars[29] = cube[12]  # cam tilt y
        # pars[36] = cube[13]  # d ff ccd
        # pars[39] = cube[14]  # ccd defocus

        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)
        # Evaluate the log-likelihood:
        # sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = np.full(len(model), 0.1)
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model['x'].values - y['posm'].values) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model['y'].values - y['posmy'].values) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 44
    path = "".join(['data/posteriors/' + date + '_full/'])
    if not os.path.exists(path):
        os.makedirs(path)
    out_file = "".join([path, 'dfit_' + str(fiber) + '_full'])

    # Run MultiNest:

    pymultinest.run(loglike, prior, n_params, n_live_points=500, outputfiles_basename=out_file, resume=True,
                    verbose=True)


    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    maxpars = output.get_mode_stats()['modes'][0]['maximum']
    print(maxpars[0])
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    mc_samples_data = pd.DataFrame(mc_samples)
    mc_samples_data.to_csv(path + 'mc_samples.tsv', sep=',', index=False)
    #samples_names = [r'Ech. G [mm$^{-1}$]', r'Coll x-tilt [deg]', r'Ech. inc. angle [deg]',
    #                r'Grism x-tilt [deg]', r'Cam x-tilt [deg]', r'Coll y-tilt [deg]',
    #                 r'Ech. $\gamma$-angle [deg]', r'Slit x-dec [mm]', r'Field flattener y-dec [mm]']





    #bestfitout = pd.DataFrame(bestfit_params['parameters'])
    #bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    #pars = parameters.load_date(fib, date)

    print('Multinest of full set of instrumental parameters for date ', date, ' done.')


def carmenes_vis_dynesty(date, fiber):
    if fiber == 'a':
        fib = 'A'
    elif fiber == 'b':
        fib = 'B'
    wsa_data, wsb_data = ws_load.load_ws_for_fit(date)
    #wsa_data = np.array(wsa_data)
    #wsb_data = np.array(wsb_data)

    pressure = env_data.get_P_at_ws(date)
    init_state = parameters.load_date(fib, date)
    init_state[-1] = pressure
    temps = env_data.get_T_at_ws(date)
    if fiber == 'a':
        ws_data = wsa_data

        spec = ws_load.spectrum_from_data(wsa_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'A', temps)
        print(len(ws_model))

        fib = 'A'

    elif fiber == 'b':
        ws_data = wsb_data
        spec = ws_load.spectrum_from_data(wsb_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'B', temps)
        fib = 'B'

    res_x = np.sqrt(np.sum((ws_data['posm'].values - ws_model['x'].values)**2) / len(ws_data))
    res_y = np.sqrt(np.sum((ws_data['posmy'].values - ws_model['y'].values) ** 2) / len(ws_data))

    print('Initial residuals, fiber = ', fiber)
    print('res_x =', res_x, ', res_y = ', res_y)
    #plt.plot(ws_data['posm'], ws_data['posm'].values - ws_model['x'].values, 'k.', zorder=10)
    #plt.plot(ws_data['posm'], ws_data['posm'].values - ws_data['posc'].values, 'b.', zorder=0, alpha=0.5)
    #plt.show()
    # We do only fiber A
    y = ws_data
    x = spec
    par_ini = init_state.copy()

    def prior(cube):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[11], delta0)  # ccd tilt z
        cube[1] = utils.transform_normal(cube[1], par_ini[9], delta0)  # echelle G
        cube[2] = utils.transform_normal(cube[2], par_ini[12], delta0)  # coll tilt x
        cube[3] = utils.transform_normal(cube[3], par_ini[21], delta0)  # echelle blaze
        cube[4] = utils.transform_normal(cube[4], par_ini[28], delta0)  # grism tilt x
        cube[5] = utils.transform_normal(cube[5], par_ini[10], delta0)  # camera x tilt
        cube[6] = utils.transform_normal(cube[6], par_ini[13], delta0)  # collimator y-tilt
        cube[7] = utils.transform_normal(cube[7], par_ini[1], delta0)  # echelle gamma angle
        cube[8] = utils.transform_normal(cube[8], par_ini[32], delta0)  # cam tilt x
        # cube[9] = utils.transform_normal(cube[9], par_ini[21], delta0)  # grm tilt x
        # cube[10] = utils.transform_normal(cube[10], par_ini[33], delta0)  # ccd ff tilt x
        # cube[11] = utils.transform_normal(cube[11], par_ini[24], delta2)  # grm apex
        # cube[12] = utils.transform_normal(cube[12], par_ini[29], delta2)  # cam tilt y
        # cube[13] = utils.transform_normal(cube[13], par_ini[36], delta2)  # d ff ccd
        # cube[14] = utils.transform_normal(cube[14], par_ini[39], delta2)  # ccd defocus
        return cube

    def loglike(cube):
        # Load parameters

        pars = parameters.load_date(fib, date)
        # print(pars[0])
        # print(cube)
        pars[11] = cube[0]
        pars[9] = cube[1]
        pars[12] = cube[2]
        pars[21] = cube[3]
        pars[28] = cube[4]
        pars[10] = cube[5]
        pars[13] = cube[6]
        pars[1] = cube[7]
        pars[32] = cube[8]
        # pars[21] = cube[9]  # grm tilt x
        # pars[33] = cube[10]  # ccd ff tilt x
        # pars[24] = cube[11]  # grm apex
        # pars[29] = cube[12]  # cam tilt y
        # pars[36] = cube[13]  # d ff ccd
        # pars[39] = cube[14]  # ccd defocus

        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)
        # Evaluate the log-likelihood:
        # sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = np.full(len(model), 0.1)
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model['x'].values - y['posm'].values) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model['y'].values - y['posmy'].values) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 9
    path = "".join(['data/posteriors/' + date + '/'])
    if not os.path.exists(path):
        os.makedirs(path)
    out_file = "".join([path, 'dyn_fit_' + str(fiber) + ''])

    # Run MultiNest:
    dsampler = dynesty.DynamicNestedSampler(
        loglike,
        prior,
        ndim=n_params
    )
    dsampler.run_nested(nlive_init=500, nlive_batch=500)
    results = dsampler.results
    samples = results['samples']
    print(samples)
    '''
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    maxpars = output.get_mode_stats()['modes'][0]['maximum']
    print(maxpars[0])
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    samples_names = [r'Ech. G [mm$^{-1}$]', r'Coll x-tilt [deg]', r'Ech. inc. angle [deg]',
                     r'Grism x-tilt [deg]', r'Cam x-tilt [deg]', r'Coll y-tilt [deg]',
                     r'Ech. $\gamma$-angle [deg]', r'Slit x-dec [mm]', r'Field flattener y-dec [mm]']


    mc_samples_data = pd.DataFrame(mc_samples)
    mc_samples_data.to_csv(path + 'mc_samples.tsv', sep=',', index=False, header=samples_names)

    bestfitout = pd.DataFrame(bestfit_params['parameters'])
    bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    pars = parameters.load_date(fib, date)

    pars[11] = bestfit_params['parameters'][0]
    pars[9] = bestfit_params['parameters'][1]
    pars[12] = bestfit_params['parameters'][2]
    pars[21] = bestfit_params['parameters'][3]
    pars[28] = bestfit_params['parameters'][4]
    pars[10] = bestfit_params['parameters'][5]
    pars[13] = bestfit_params['parameters'][6]
    pars[1] = bestfit_params['parameters'][7]
    pars[32] = bestfit_params['parameters'][8]
    #pars[42] = maxpars[0]
    #pars[11] = maxpars[1]
    #pars[9] = maxpars[2]
    #pars[12] = maxpars[3]
    #pars[21] = maxpars[4]
    #pars[28] = maxpars[5]
    #pars[10] = maxpars[6]
    #pars[13] = maxpars[7]
    #pars[1] = maxpars[8]
    #parameters.write_sa(pars, date, fiber)
    #fit_sa.moes_carmenes_vis_old(date)
    do_cornerplot.do_plot(date)
    # figure = corner.corner(mc_samples)  # , labels=posterior_names)
    # Get output:
    #output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    #bestfit_params = output.get_best_fit()
    #mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    #outdir = path
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    #bestfitout = pd.DataFrame(bestfit_params['parameters'])
    #bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    #figure = corner.corner(mc_samples)  # , labels=posterior_names)
    #plt.tight_layout()
    #plt.savefig(outdir + 'parameters_cornerplot_'+str(fiber)+'.png')
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    '''
    print('dynesty of instrumental parameters for date ', date, ' done.')


def nzp_res_dd_offset():
    out_path = 'output_files/'
    mavc_path = '/home/eduspec/Documentos/CARMENES/CARMENES_data/CARM_VIS_AVC_201017_corrected/avcn/'

    # Load data
    all_data = pd.read_csv(out_path + 'all_avc_dd_update.dat', sep=',')
    all_data = all_data.dropna()
    print(all_data.columns)
    bjdref = 2458000

    nzp = all_data['nzp_obs']
    nzp_err = all_data['e_nzp_obs']
    nzp_mean = np.mean(nzp)
    ddres = all_data['res_dd_rv_mean']
    # Define the prior (you have to transform your parameters, that come from the unit cube,
    # to the prior you want):
    res = nzp - ddres
    print(np.std(res), np.sqrt(np.mean(res) ** 2))

    def prior(cube, ndim, nparams):
        cube[0] = utils.transform_uniform(cube[0], -5, 5)  # offset

    def loglike(cube, ndim, nparams):
        # Load parameters
        pars = [0.]
        pars[0] = cube[0]  # offset
        model = (ddres + pars[0])
        sigma_fit_y = nzp_err
        ndata = len(nzp)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_y ** 2).sum() + (
                -0.5 * ((nzp - model) / sigma_fit_y) ** 2).sum()
        return loglikelihood

    n_params = 15
    path = 'ns_moes/'
    if not os.path.exists(path):
        os.makedirs(path)
    out_file = 'ns_moes/nzp_dd_residuals_offset_'

    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points=5000, outputfiles_basename=out_file, resume=True,
                    verbose=False)

    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    offset = bestfit_params['parameters'][0]

    # ERROR PENDING

    return offset
    # import corner
    # posterior_names = []
    # pars_index = [42,11,12,9,13,10,35,17,28,21,33,24,29,36,39]
    # new_pars = init_state
    # i = 0
    # for par in pars_index:
    #    posterior_names.append(parameters.get_name(par))
    #    new_pars[par] = bestfit_params['parameters'][i]
    #    print(parameters.get_name(par), init_state[par], new_pars[par], bestfit_params['parameters'][i])
    #    i += 1

    # first_time = True
    # for i in range(n_params):
    #    if first_time:
    #        posterior_data = output.posteriors['posterior_samples'][i]
    #        first_time = False
    #    else:
    #        posterior_data  = np.vstack((posterior_data, results.posteriors['posterior_samples'][i]))

    # posterior_data = posterior_data.T
    # figure = corner.corner(mc_samples, labels = posterior_names)
    # plt.savefig('plots/moes_ins_corner_'+fiber+'_'+date+'.png')


def carmenes_vis_dynesty_full(date, fiber):
    if fiber == 'a':
        fib = 'A'
    elif fiber == 'b':
        fib = 'B'
    wsa_data, wsb_data = ws_load.load_ws_for_fit(date)
    #wsa_data = np.array(wsa_data)
    #wsb_data = np.array(wsb_data)

    pressure = env_data.get_P_at_ws(date)
    init_state = parameters.load_date(fib, date)
    init_state[-1] = pressure
    temps = env_data.get_T_at_ws(date)
    if fiber == 'a':
        ws_data = wsa_data

        spec = ws_load.spectrum_from_data(wsa_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'A', temps)
        print(len(ws_model))

        fib = 'A'

    elif fiber == 'b':
        ws_data = wsb_data
        spec = ws_load.spectrum_from_data(wsb_data)
        ws_model = vis_spectrometer.tracing(spec, init_state, 'B', temps)
        fib = 'B'

    res_x = np.sqrt(np.sum((ws_data['posm'].values - ws_model['x'].values)**2) / len(ws_data))
    res_y = np.sqrt(np.sum((ws_data['posmy'].values - ws_model['y'].values) ** 2) / len(ws_data))

    print('Initial residuals, fiber = ', fiber)
    print('res_x =', res_x, ', res_y = ', res_y)
    #plt.plot(ws_data['posm'], ws_data['posm'].values - ws_model['x'].values, 'k.', zorder=10)
    #plt.plot(ws_data['posm'], ws_data['posm'].values - ws_data['posc'].values, 'b.', zorder=0, alpha=0.5)
    #plt.show()
    # We do only fiber A
    y = ws_data
    x = spec
    par_ini = init_state.copy()

    def prior(cube):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[0], delta0)
        cube[1] = utils.transform_normal(cube[1], par_ini[1], delta0)
        cube[2] = utils.transform_normal(cube[2], par_ini[2], delta0)
        cube[3] = utils.transform_normal(cube[3], par_ini[3], delta0)
        cube[4] = utils.transform_normal(cube[4], par_ini[4], delta0)
        cube[5] = utils.transform_normal(cube[5], par_ini[5], delta0)
        cube[6] = utils.transform_normal(cube[6], par_ini[6], delta0)
        cube[7] = utils.transform_normal(cube[7], par_ini[7], delta0)
        cube[8] = utils.transform_normal(cube[8], par_ini[8], delta0)
        cube[9] = utils.transform_normal(cube[9], par_ini[9], delta0)
        cube[10] = utils.transform_normal(cube[10], par_ini[10], delta0)
        cube[11] = utils.transform_normal(cube[11], par_ini[11], delta0)
        cube[12] = utils.transform_normal(cube[12], par_ini[12], delta0)
        cube[13] = utils.transform_normal(cube[13], par_ini[13], delta0)
        cube[14] = utils.transform_normal(cube[14], par_ini[14], delta0)
        cube[15] = utils.transform_normal(cube[15], par_ini[15], delta0)
        cube[16] = utils.transform_normal(cube[16], par_ini[16], delta0)
        cube[17] = utils.transform_normal(cube[17], par_ini[17], delta0)
        cube[18] = utils.transform_normal(cube[18], par_ini[18], delta0)
        cube[19] = utils.transform_normal(cube[19], par_ini[19], delta0)
        cube[20] = utils.transform_normal(cube[20], par_ini[20], delta0)
        cube[21] = utils.transform_normal(cube[21], par_ini[21], delta0)
        cube[22] = utils.transform_normal(cube[22], par_ini[22], delta0)
        cube[23] = utils.transform_normal(cube[23], par_ini[23], delta0)
        cube[24] = utils.transform_normal(cube[24], par_ini[24], delta0)
        cube[25] = utils.transform_normal(cube[25], par_ini[25], delta0)
        cube[26] = utils.transform_normal(cube[26], par_ini[26], delta0)
        cube[27] = utils.transform_normal(cube[27], par_ini[27], delta0)
        cube[28] = utils.transform_normal(cube[28], par_ini[28], delta0)
        cube[29] = utils.transform_normal(cube[29], par_ini[29], delta0)
        cube[30] = utils.transform_normal(cube[30], par_ini[30], delta0)
        cube[31] = utils.transform_normal(cube[31], par_ini[31], delta0)
        cube[32] = utils.transform_normal(cube[32], par_ini[32], delta0)
        cube[33] = utils.transform_normal(cube[33], par_ini[33], delta0)
        cube[34] = utils.transform_normal(cube[34], par_ini[34], delta0)
        cube[35] = utils.transform_normal(cube[35], par_ini[35], delta0)
        cube[36] = utils.transform_normal(cube[36], par_ini[36], delta0)
        cube[37] = utils.transform_normal(cube[37], par_ini[37], delta0)
        cube[38] = utils.transform_normal(cube[38], par_ini[38], delta0)
        cube[39] = utils.transform_normal(cube[39], par_ini[39], delta0)
        cube[40] = utils.transform_normal(cube[40], par_ini[40], delta0)
        cube[41] = utils.transform_normal(cube[41], par_ini[41], delta0)
        cube[42] = utils.transform_normal(cube[42], par_ini[42], delta0)
        cube[43] = utils.transform_normal(cube[43], par_ini[43], delta0)
        # cube[9] = utils.transform_normal(cube[9], par_ini[21], delta0)  # grm tilt x
        # cube[10] = utils.transform_normal(cube[10], par_ini[33], delta0)  # ccd ff tilt x
        # cube[11] = utils.transform_normal(cube[11], par_ini[24], delta2)  # grm apex
        # cube[12] = utils.transform_normal(cube[12], par_ini[29], delta2)  # cam tilt y
        # cube[13] = utils.transform_normal(cube[13], par_ini[36], delta2)  # d ff ccd
        # cube[14] = utils.transform_normal(cube[14], par_ini[39], delta2)  # ccd defocus
        return cube

    def loglike(cube):
        # Load parameters
        pars = parameters.load_date(fib, date)
        # print(pars[0])
        # print(cube)
        pars[0] = cube[0]
        pars[1] = cube[1]
        pars[2] = cube[2]
        pars[3] = cube[3]
        pars[4] = cube[4]
        pars[5] = cube[5]
        pars[6] = cube[6]
        pars[7] = cube[7]
        pars[8] = cube[8]
        pars[9] = cube[9]
        pars[10] = cube[10]
        pars[11] = cube[11]
        pars[12] = cube[12]
        pars[13] = cube[13]
        pars[14] = cube[14]
        pars[15] = cube[15]
        pars[16] = cube[16]
        pars[17] = cube[17]
        pars[18] = cube[18]
        pars[19] = cube[19]
        pars[20] = cube[20]
        pars[21] = cube[21]
        pars[22] = cube[22]
        pars[23] = cube[23]
        pars[24] = cube[24]
        pars[25] = cube[25]
        pars[26] = cube[26]
        pars[27] = cube[27]
        pars[28] = cube[28]
        pars[29] = cube[29]
        pars[30] = cube[30]
        pars[31] = cube[31]
        pars[32] = cube[32]
        pars[33] = cube[33]
        pars[34] = cube[34]
        pars[35] = cube[35]
        pars[36] = cube[36]
        pars[37] = cube[37]
        pars[38] = cube[38]
        pars[39] = cube[39]
        pars[40] = cube[40]
        pars[41] = cube[41]
        pars[42] = cube[42]
        pars[43] = cube[43]

        # pars[21] = cube[9]  # grm tilt x
        # pars[33] = cube[10]  # ccd ff tilt x
        # pars[24] = cube[11]  # grm apex
        # pars[29] = cube[12]  # cam tilt y
        # pars[36] = cube[13]  # d ff ccd
        # pars[39] = cube[14]  # ccd defocus

        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)
        # Evaluate the log-likelihood:
        # sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = np.full(len(model), 0.1)
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model['x'].values - y['posm'].values) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model['y'].values - y['posmy'].values) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 44
    path = "".join(['data/posteriors/' + date + '_full/'])
    if not os.path.exists(path):
        os.makedirs(path)
    out_file = "".join([path, 'dyn_fit_' + str(fiber) + ''])

    # Run MultiNest:
    dsampler = dynesty.DynamicNestedSampler(
        loglike,
        prior,
        ndim=n_params
    )
    dsampler.run_nested(nlive_init=500, nlive_batch=500)
    results = dsampler.results
    samples = results['samples']
    mc_samples_data = pd.DataFrame(samples)
    mc_samples_data.to_csv(path + 'dyn_mc_samples.tsv', sep=',', index=False)
    '''
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    maxpars = output.get_mode_stats()['modes'][0]['maximum']
    print(maxpars[0])
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    samples_names = [r'Ech. G [mm$^{-1}$]', r'Coll x-tilt [deg]', r'Ech. inc. angle [deg]',
                     r'Grism x-tilt [deg]', r'Cam x-tilt [deg]', r'Coll y-tilt [deg]',
                     r'Ech. $\gamma$-angle [deg]', r'Slit x-dec [mm]', r'Field flattener y-dec [mm]']


    mc_samples_data = pd.DataFrame(mc_samples)
    mc_samples_data.to_csv(path + 'mc_samples.tsv', sep=',', index=False, header=samples_names)

    bestfitout = pd.DataFrame(bestfit_params['parameters'])
    bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    pars = parameters.load_date(fib, date)

    pars[11] = bestfit_params['parameters'][0]
    pars[9] = bestfit_params['parameters'][1]
    pars[12] = bestfit_params['parameters'][2]
    pars[21] = bestfit_params['parameters'][3]
    pars[28] = bestfit_params['parameters'][4]
    pars[10] = bestfit_params['parameters'][5]
    pars[13] = bestfit_params['parameters'][6]
    pars[1] = bestfit_params['parameters'][7]
    pars[32] = bestfit_params['parameters'][8]
    #pars[42] = maxpars[0]
    #pars[11] = maxpars[1]
    #pars[9] = maxpars[2]
    #pars[12] = maxpars[3]
    #pars[21] = maxpars[4]
    #pars[28] = maxpars[5]
    #pars[10] = maxpars[6]
    #pars[13] = maxpars[7]
    #pars[1] = maxpars[8]
    #parameters.write_sa(pars, date, fiber)
    #fit_sa.moes_carmenes_vis_old(date)
    do_cornerplot.do_plot(date)
    # figure = corner.corner(mc_samples)  # , labels=posterior_names)
    # Get output:
    #output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    #bestfit_params = output.get_best_fit()
    #mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    #outdir = path
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    #bestfitout = pd.DataFrame(bestfit_params['parameters'])
    #bestfitout.to_csv(path + 'best_fit_params_'+str(fiber)+'.tsv', index=False, header=False)
    #samplesout = pd.DataFrame(mc_samples)
    #samplesout.to_csv(path + 'mc_samples_'+str(fiber)+'.tsv', index=False, header=False)
    #figure = corner.corner(mc_samples)  # , labels=posterior_names)
    #plt.tight_layout()
    #plt.savefig(outdir + 'parameters_cornerplot_'+str(fiber)+'.png')
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    '''
    print('dynesty of full instrumental parameters for date ', date, ' done.')


def do_priors_full_model():
    init_state = parameters.load_date('A', '2017-10-20')
    for i in range(len(init_state)):
        print('cube['+str(int(i))+'] = utils.transform_normal(cube['+str(int(i))+'], par_ini['+str(int(i))+'], delta0)')

def do_loglikes_full_model():
    init_state = parameters.load_date('A', '2017-10-20')
    for i in range(len(init_state)):
        print('pars['+str(int(i))+'] = cube['+str(int(i))+']')



if __name__ == '__main__':
    import optimization
    date = '2017-10-20'
    #rewrite_best_fit_params(date)
    #optimization.simulated_annealing_fit_date(date, 'A')
    #carmenes_vis_multinest_full(date, 'a')
    carmenes_vis_dynesty_full(date, 'a')
    #do_priors_full_model()
    #do_loglikes_full_model()


