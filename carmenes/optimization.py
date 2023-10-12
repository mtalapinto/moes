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


class WavelengthSolutionFit(Annealer):
    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, ws_data, spectrum, temps, fib):
        self.ws_data = ws_data
        self.spectrum = spectrum
        self.temps = temps
        self.fib = fib
        super(WavelengthSolutionFit, self).__init__(state)  # important!

    def move(self):
        # Steps on the parameters space
        delta_all = 1e-7
        delta_0 = 5e-7
        delta_1 = 5e-7
        delta_2 = 5e-7  # 1e-4
        delta_3 = 1e-7  # 1e-2

        if self.fib == 'a':
            self.state = [np.random.normal(self.state[0], delta_all),  # slit_dec_x_a
                          np.random.normal(self.state[1], delta_0),  # slit_dec_y_a
                          np.random.normal(self.state[2], 1e-99),  # slit_dec_x_b
                          np.random.normal(self.state[3], 1e-99),  # slit_dec_y_b
                          np.random.normal(self.state[4], delta_all),  # slit_defocus
                          np.random.normal(self.state[5], delta_all),  # slit_tilt_x
                          np.random.normal(self.state[6], delta_all),  # slit_tilt_y
                          np.random.normal(self.state[7], delta_all),  # slit_tilt_z
                          np.random.normal(self.state[8], delta_all),  # d_slit_col
                          np.random.normal(self.state[9], delta_0),  # coll_tilt_x
                          np.random.normal(self.state[10], delta_0),  # coll_tilt_y
                          np.random.normal(self.state[11], delta_0),  # ech_G
                          np.random.normal(self.state[12], delta_0),  # ech_blaze
                          np.random.normal(self.state[13], delta_0),  # ech_gamma
                          np.random.normal(self.state[14], delta_0),  # ech_z
                          np.random.normal(self.state[15], delta_all),  # d_col_trf
                          np.random.normal(self.state[16], delta_all),  # trf_mirror_tilt_x
                          np.random.normal(self.state[17], delta_all),  # trf_mirror_tilt_y
                          np.random.normal(self.state[18], delta_all),  # d_col_grm
                          np.random.normal(self.state[19], delta_all),  # grism_dec_x
                          np.random.normal(self.state[20], delta_all),  # grism_dec_y
                          np.random.normal(self.state[21], delta_0),  # grm_tilt_x
                          np.random.normal(self.state[22], delta_all),  # grm_tilt_y
                          np.random.normal(self.state[23], delta_all),  # grm_G
                          np.random.normal(self.state[24], delta_all),  # grm_apex
                          np.random.normal(self.state[25], delta_all),  # d_grm_cam
                          np.random.normal(self.state[26], delta_all),  # cam_dec_x
                          np.random.normal(self.state[27], delta_all),  # cam_dec_y
                          np.random.normal(self.state[28], delta_0),  # cam_tilt_x
                          np.random.normal(self.state[29], delta_all),  # cam_tilt_y
                          np.random.normal(self.state[30], delta_all),  # d_cam_ff
                          np.random.normal(self.state[31], delta_all),  # ccd_ff_dec_x
                          np.random.normal(self.state[32], delta_all),  # ccd_ff_dec_y
                          np.random.normal(self.state[33], delta_all),  # ccd_ff_tilt_x
                          np.random.normal(self.state[34], delta_all),  # ccd_ff_tilt_y
                          np.random.normal(self.state[35], delta_all),  # ccd_ff_tilt_z
                          np.random.normal(self.state[36], delta_all),  # d_ff_ccd
                          np.random.normal(self.state[37], delta_all),  # ccd_dec_x
                          np.random.normal(self.state[38], delta_all),  # ccd_dec_y
                          np.random.normal(self.state[39], delta_all),  # ccd_defocus
                          np.random.normal(self.state[40], delta_all),  # ccd_tilt_x
                          np.random.normal(self.state[41], delta_all),  # ccd_tilt_y
                          np.random.normal(self.state[42], delta_1),  # ccd_tilt_z
                          np.random.normal(self.state[43], 1e-99),  # p
                          ]
        else:
            self.state = [np.random.normal(self.state[0], 1e-99),  # slit_dec_x_a
                          np.random.normal(self.state[1], 1e-99),  # slit_dec_y_a
                          np.random.normal(self.state[2], 1e-95),  # slit_dec_x_b
                          np.random.normal(self.state[3], 1e-95),  # slit_dec_y_b
                          np.random.normal(self.state[4], delta_all),  # slit_defocus
                          np.random.normal(self.state[5], delta_all),  # slit_tilt_x
                          np.random.normal(self.state[6], delta_all),  # slit_tilt_y
                          np.random.normal(self.state[7], delta_all),  # slit_tilt_z
                          np.random.normal(self.state[8], delta_all),  # d_slit_col
                          np.random.normal(self.state[9], delta_all),  # coll_tilt_x
                          np.random.normal(self.state[10], delta_all),  # coll_tilt_y
                          np.random.normal(self.state[11], delta_1),  # ech_G
                          np.random.normal(self.state[12], delta_2),  # ech_blaze
                          np.random.normal(self.state[13], delta_0),  # ech_gamma
                          np.random.normal(self.state[14], delta_0),  # ech_z
                          np.random.normal(self.state[15], delta_2),  # d_col_trf
                          np.random.normal(self.state[16], delta_all),  # trf_mirror_tilt_x
                          np.random.normal(self.state[17], delta_all),  # trf_mirror_tilt_y
                          np.random.normal(self.state[18], delta_all),  # d_col_grm
                          np.random.normal(self.state[19], delta_all),  # grism_dec_x
                          np.random.normal(self.state[20], delta_all),  # grism_dec_y
                          np.random.normal(self.state[21], delta_0),  # grm_tilt_x
                          np.random.normal(self.state[22], delta_0),  # grm_tilt_y
                          np.random.normal(self.state[23], delta_0),  # grm_G
                          np.random.normal(self.state[24], delta_all),  # grm_apex
                          np.random.normal(self.state[25], delta_2),  # d_grm_cam
                          np.random.normal(self.state[26], delta_all),  # cam_dec_x
                          np.random.normal(self.state[27], delta_all),  # cam_dec_y
                          np.random.normal(self.state[28], delta_all),  # cam_tilt_x
                          np.random.normal(self.state[29], delta_all),  # cam_tilt_y
                          np.random.normal(self.state[30], 5 * delta_all),  # d_cam_ff
                          np.random.normal(self.state[31], delta_0),  # ccd_ff_dec_x
                          np.random.normal(self.state[32], delta_all),  # ccd_ff_dec_y
                          np.random.normal(self.state[33], delta_all),  # ccd_ff_tilt_x
                          np.random.normal(self.state[34], delta_0),  # ccd_ff_tilt_y
                          np.random.normal(self.state[35], delta_0),  # ccd_ff_tilt_z
                          np.random.normal(self.state[36], delta_all),  # d_ff_ccd
                          np.random.normal(self.state[37], delta_2),  # ccd_dec_x
                          np.random.normal(self.state[38], delta_2),  # ccd_dec_y
                          np.random.normal(self.state[39], delta_3),  # ccd_defocus
                          np.random.normal(self.state[40], delta_all),  # ccd_tilt_x
                          np.random.normal(self.state[41], delta_all),  # ccd_tilt_y
                          np.random.normal(self.state[42], delta_1),  # ccd_tilt_z
                          np.random.normal(self.state[43], 1e-99),  # p
                          ]

    def energy(self):
        # difference between model and data
        if self.fib == 'a':
            ws_mod = vis_spectrometer.tracing(self.spectrum, self.state, 'A', self.temps)
        else:
            ws_mod = vis_spectrometer.tracing(self.spectrum, self.state, 'B', self.temps)

        #chi2x = np.sum((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2 / self.ws_data[:, 4])
        #chi2x = np.sum((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2 / len(self.state))
        chi2x = np.sqrt(np.mean((self.ws_data['posm'].values - ws_mod['x'].values) ** 2))
        #chi2x = np.sqrt(np.mean((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2))
        chi2y = np.sqrt(np.mean((self.ws_data['posmy'].values - ws_mod['y'].values) ** 2))
        #chi2y = np.sum((self.ws_data[:, 5] - ws_mod[:, 3]) ** 2 / self.ws_data[:, 4])
        #rchi2x = chi2x / len(self.state)
        #rchi2y = chi2y / len(self.state)
        rchi2 = np.sqrt(chi2x ** 2 + chi2y ** 2)
        e = rchi2
        #e = np.sqrt(np.mean((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2))
        return e


def sine(x, c1, c2, c3, c4):
    #func = np.abs(x)
    amp1 = c1 * x ** 2
    per1 = np.sin(c2 * x + c4)
    return amp1 * per1 + c3


def run_instrument_model():
    wsa_data, wsb_data = ws_load.carmenes_vis_ws()
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state_a = parameters.load_sa('a')
    init_state_b = parameters.load_sa('b')
    temps = env_data.get_temps()
    pressure = env_data.get_p()
    init_state_a[-1] = pressure
    init_state_b[-1] = pressure
    wsa = vis_spectrometer.tracing(spec_a, init_state_a, 'a', temps)
    wsb = vis_spectrometer.tracing(spec_b, init_state_b, 'b', temps)
    return wsa, wsb


def load_aberrations(ws, fib):
    #print(ws)
    ws_model = aberration_corrections.chromatic_model_load_date(date, ws, fib)
    #ws_model = aberration_corrections.optical_model_load_date(date, ws_model, fib)

    return ws_model


def nested_sampling_fit(fiber):
    wsa_data, wsb_data = ws_load.carmenes_vis_ws()
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state = parameters.load_sa(fiber)
    temps = env_data.get_temps()
    pressure = env_data.get_p()
    init_state[-1] = pressure

    if fiber == 'a':
        wsa = vis_spectrometer.tracing(spec_a, init_state, 'A', temps)
        ws_data = wsa_data.copy()
        spec = spec_a.copy()
        fib = 'A'
    elif fiber == 'b':
        wsb = vis_spectrometer.tracing(spec_b, init_state, 'B', temps)
        ws_data = wsb_data.copy()
        spec = spec_b.copy()
        fib = 'B'

    # We do only fiber A
    y = ws_data
    x = spec
    par_ini = init_state.copy()

    def prior(cube):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[42], 7.5e-7)  # ccd tilt z
        cube[1] = utils.transform_normal(cube[1], par_ini[11], delta0)  # echelle G
        cube[2] = utils.transform_normal(cube[2], par_ini[9], delta0)  # coll tilt x
        cube[3] = utils.transform_normal(cube[3], par_ini[12], delta0)  # echelle blaze
        cube[4] = utils.transform_normal(cube[4], par_ini[21], delta0)  # grism tilt x
        cube[5] = utils.transform_normal(cube[5], par_ini[28], delta0)  # camera x tilt
        cube[6] = utils.transform_normal(cube[6], par_ini[10], delta0)  # collimator y-tilt
        cube[7] = utils.transform_normal(cube[7], par_ini[13], delta0)  # echelle gamma angle
        cube[8] = utils.transform_normal(cube[8], par_ini[3], delta0)  # cam tilt x
        # cube[9] = utils.transform_normal(cube[9], par_ini[21], delta0)  # grm tilt x
        # cube[10] = utils.transform_normal(cube[10], par_ini[33], delta0)  # ccd ff tilt x
        # cube[11] = utils.transform_normal(cube[11], par_ini[24], delta2)  # grm apex
        # cube[12] = utils.transform_normal(cube[12], par_ini[29], delta2)  # cam tilt y
        # cube[13] = utils.transform_normal(cube[13], par_ini[36], delta2)  # d ff ccd
        # cube[14] = utils.transform_normal(cube[14], par_ini[39], delta2)  # ccd defocus
        return cube

    def loglike(cube):
        # Load parameters

        pars = parameters.load_sa(fiber)
        # print(pars[0])
        # print(cube)
        pars[42] = cube[0]  # ccd tilt z
        pars[11] = cube[1]  # echelle G
        pars[9] = cube[2]  # echelle blaze
        pars[12] = cube[3]  # coll tilt x
        pars[21] = cube[4]  # echelle gamma
        pars[28] = cube[5]  # coll tilt y
        pars[10] = cube[6]  # ccd_ff_tilt_z
        pars[13] = cube[7]  # trf mirror tilt y
        pars[3] = cube[8]  # cam tilt x
        # pars[21] = cube[9]  # grm tilt x
        # pars[33] = cube[10]  # ccd ff tilt x
        # pars[24] = cube[11]  # grm apex
        # pars[29] = cube[12]  # cam tilt y
        # pars[36] = cube[13]  # d ff ccd
        # pars[39] = cube[14]  # ccd defocus

        if len(pars) < 43:
            print('chafa')
        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)

        # Evaluate the log-likelihood:
        # sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = y[:, 4]
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model[:, 2] - y[:, 3]) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model[:, 3] - y[:, 5]) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 9
    path = "".join(['data/instrumental_parameters/dynesty_results/'])

    if not os.path.exists(path):
        os.makedirs(path)

    # Run dynesty:
    dsampler = dynesty.DynamicNestedSampler(
        loglike,
        prior,
        ndim=n_params)
    dsampler.run_nested(nlive_init=300, nlive_batch=300)
    results = dsampler.results
    out = pd.DataFrame(results)
    out.to_csv(path+'carmenes_vis_dyn_samples.tsv', sep=',', index=False)
    print('Samples recorded to file.')


def nested_sampling_fit_date(fiber, date):
    wsa_data, wsb_data = ws_load.carmenes_vis_ws()
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state = parameters.load_sa(fiber)
    temps = env_data.get_temps()
    pressure = env_data.get_p()
    init_state[-1] = pressure

    if fiber == 'a':
        wsa = vis_spectrometer.tracing(spec_a, init_state, 'A', temps)
        ws_data = wsa_data.copy()
        spec = spec_a.copy()
        fib = 'A'
    elif fiber == 'b':
        wsb = vis_spectrometer.tracing(spec_b, init_state, 'B', temps)
        ws_data = wsb_data.copy()
        spec = spec_b.copy()
        fib = 'B'

    # We do only fiber A
    y = ws_data
    x = spec
    par_ini = init_state.copy()

    def prior(cube):
        # Prior on RAMSES parameters, sorted by importance
        delta0 = 3.5e-6
        delta1 = 1.e-6
        delta2 = 5.e-5
        cube[0] = utils.transform_normal(cube[0], par_ini[42], 7.5e-7)  # ccd tilt z
        cube[1] = utils.transform_normal(cube[1], par_ini[11], delta0)  # echelle G
        cube[2] = utils.transform_normal(cube[2], par_ini[9], delta0)  # coll tilt x
        cube[3] = utils.transform_normal(cube[3], par_ini[12], delta0)  # echelle blaze
        cube[4] = utils.transform_normal(cube[4], par_ini[21], delta0)  # grism tilt x
        cube[5] = utils.transform_normal(cube[5], par_ini[28], delta0)  # camera x tilt
        cube[6] = utils.transform_normal(cube[6], par_ini[10], delta0)  # collimator y-tilt
        cube[7] = utils.transform_normal(cube[7], par_ini[13], delta0)  # echelle gamma angle
        cube[8] = utils.transform_normal(cube[8], par_ini[3], delta0)  # cam tilt x
        # cube[9] = utils.transform_normal(cube[9], par_ini[21], delta0)  # grm tilt x
        # cube[10] = utils.transform_normal(cube[10], par_ini[33], delta0)  # ccd ff tilt x
        # cube[11] = utils.transform_normal(cube[11], par_ini[24], delta2)  # grm apex
        # cube[12] = utils.transform_normal(cube[12], par_ini[29], delta2)  # cam tilt y
        # cube[13] = utils.transform_normal(cube[13], par_ini[36], delta2)  # d ff ccd
        # cube[14] = utils.transform_normal(cube[14], par_ini[39], delta2)  # ccd defocus
        return cube

    def loglike(cube):
        # Load parameters

        pars = parameters.load_sa(fiber)
        # print(pars[0])
        # print(cube)
        pars[42] = cube[0]  # ccd tilt z
        pars[11] = cube[1]  # echelle G
        pars[9] = cube[2]  # echelle blaze
        pars[12] = cube[3]  # coll tilt x
        pars[21] = cube[4]  # echelle gamma
        pars[28] = cube[5]  # coll tilt y
        pars[10] = cube[6]  # ccd_ff_tilt_z
        pars[13] = cube[7]  # trf mirror tilt y
        pars[3] = cube[8]  # cam tilt x
        # pars[21] = cube[9]  # grm tilt x
        # pars[33] = cube[10]  # ccd ff tilt x
        # pars[24] = cube[11]  # grm apex
        # pars[29] = cube[12]  # cam tilt y
        # pars[36] = cube[13]  # d ff ccd
        # pars[39] = cube[14]  # ccd defocus

        if len(pars) < 43:
            print('chafa')
        # Generate model:
        model = vis_spectrometer.tracing(x, pars, fib, temps)

        # Evaluate the log-likelihood:
        # sigma_fit_x = np.full(len(y), y[:, 4])
        sigma_fit_x = y[:, 4]
        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + \
                        (-0.5 * ((model[:, 2] - y[:, 3]) / sigma_fit_x) ** 2).sum() + \
                        (-0.5 * ((model[:, 3] - y[:, 5]) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 9
    path = "".join(['data/instrumental_parameters/dynesty_results/'])

    if not os.path.exists(path):
        os.makedirs(path)

    # Run dynesty:
    dsampler = dynesty.DynamicNestedSampler(
        loglike,
        prior,
        ndim=n_params)
    dsampler.run_nested(nlive_init=300, nlive_batch=300)
    results = dsampler.results
    out = pd.DataFrame(results)
    out.to_csv(path+'carmenes_vis_dyn_samples.tsv', sep=',', index=False)
    print('Samples recorded to file.')


def simulated_annealing_fit():
    print('Running simulated annealing optimization\n')
    wsa_data, wsb_data = ws_load.carmenes_vis_ws()
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    print('CARMENES VIS data loaded.')
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state_a = parameters.load('a')
    init_state_b = parameters.load('b')
    temps = env_data.get_temps()
    pressure = env_data.get_p()
    init_state_a[-1] = pressure
    init_state_b[-1] = pressure
    print('MOES parameters loaded.\n')
    # wsa = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    # wsb = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)

    wsa_fit = WavelengthSolutionFit(init_state_a, wsa_data, spec_a, temps, 'a')
    wsa_fit.steps = 2500
    wsa_fit.Tmax = 12500
    wsa_fit.Tmin = 1e-10

    state_a, e_a = wsa_fit.anneal()
    print('\n')
    wsa_model = vis_spectrometer.tracing(spec_a, state_a, 'A', temps)
    resxa_aux = np.sqrt(np.mean((wsa_data[:, 3] - wsa_model[:, 2]) ** 2))
    resya_aux = np.sqrt(np.mean((wsa_data[:, 5] - wsa_model[:, 3]) ** 2))
    parameters.write(state_a, 'a')
    print('Post-fit residuals rms for fiber A, in x = ', resxa_aux, 'pix ', ' in y = ', resya_aux, ' pix')

    print('Fiber B')
    # Simulated annealing routine for fiber B
    wsb_fit = WavelengthSolutionFit(init_state_b, wsb_data, spec_b, temps, 'b')
    wsb_fit.steps = 2500
    wsb_fit.Tmax = 12500
    wsb_fit.Tmin = 1e-10

    state_b, e_b = wsb_fit.anneal()
    wsb_model = vis_spectrometer.tracing(spec_b, state_b, 'B', temps)
    resxb_aux = np.sqrt(np.mean((wsb_data[:, 3] - wsb_model[:, 2]) ** 2))
    resyb_aux = np.sqrt(np.mean((wsb_data[:, 5] - wsb_model[:, 3]) ** 2))
    parameters.write(state_b, 'b')
    print('Post-fit residuals rms for fiber B, in x = ', resxb_aux, 'pix ', ' in y = ', resyb_aux, ' pix')


def simulated_annealing_fit_date(date, fib):
    print('Running simulated annealing optimization\n')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    kin = 'hcl'
    print('Loading CARMENES data...', date),
    data = ws_load.read_ws(date, kin, fib)
    print('loaded.')
    print('Loading MOES model with current date...', date),
    spec = ws_load.spectrum_from_data(data)
    # init_state = parameters.load_date(fib, date)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    print(temps)
    print(pressure)
    init_state[-1] = pressure
    model = vis_spectrometer.tracing(spec, init_state, fib, temps)
    residualsx = data['posm'].values - model['x'].values
    residualsy = data['posmy'].values - model['y'].values
    rmsx = np.sqrt(np.sum(residualsx ** 2) / len(residualsx))
    rmsy = np.sqrt(np.sum(residualsy ** 2) / len(residualsy))
    print('Pre-fit residuals rms in x = ', rmsx, ', in y = ', rmsy)

    #plt.plot(data['posm'].values, residualsx, 'k.')
    #plt.show()
    #plt.clf()

    print('MOES parameters loaded.\n')
    # wsa = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    # wsb = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)

    ws_fit = WavelengthSolutionFit(init_state, data, spec, temps, fiber)
    ws_fit.steps = 2500
    ws_fit.Tmax = 12500
    ws_fit.Tmin = 1e-10
    state, energy_a = ws_fit.anneal()

    print('\n')
    ws_model = vis_spectrometer.tracing(spec, state, fib, temps)
    residualsx = data['posm'].values - ws_model['x'].values
    residualsy = data['posmy'].values - ws_model['y'].values
    rmsx = np.sqrt(np.sum(residualsx ** 2) / len(residualsx))
    rmsy = np.sqrt(np.sum(residualsy ** 2) / len(residualsy))
    #plt.plot(data['posm'].values, residualsx, 'k.')
    #plt.show()
    #plt.clf()
    parameters.write_date(state, fiber, date)
    print('Parameters file for ', date, ' written.')
    print('Post-fit residuals rms for fiber '+fib+', in x = ', rmsx, 'pix ', ' in y = ', rmsy, ' pix')

    # We write the parameters file for the next date
    datews = Time(date+'T00:00:00.0', format='isot')
    jddate = datews.jd
    k = 1
    parsdir = 'data/instrumental_parameters_timeseries/'
    while k < 7:
        jdnext = jddate + k
        datenext = Time(jdnext, format='jd')
        datenext = datenext.isot
        datenext = datenext[:10]
        nextdir = parsdir + datenext
        if os.path.exists(nextdir):
            parameters.write_date(state, fiber, datenext)
            print('Parameters file for next date ', datenext, ' written.')
            k = 7
        else:
            k += 1


def fit_instrument_model(style):
    if style == 'nested-sampling':
        nested_sampling_fit('a')
        nested_sampling_fit('b')

    elif style == 'simulated-annealing':
        simulated_annealing_fit()


def fit_instrument_model_date(style, date):
    if style == 'nested-sampling':
        nested_sampling_fit('a')
        nested_sampling_fit('b')

    elif style == 'simulated-annealing':
        simulated_annealing_fit()


def fit_chromatic_aberrations():
    print('Calculating chromatic aberration coefficients')
    print('Loading CARMENES data...'),
    wsa_data, wsb_data = ws_load.carmenes_vis_ws()
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    print('done.')
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state_a = parameters.load('a')
    init_state_b = parameters.load('b')
    print('MOES parameters loaded.')
    temps = env_data.get_temps()
    pressure = env_data.get_p()
    init_state_a[-1] = pressure
    init_state_b[-1] = pressure
    print('Tracing rays... '),
    wsa = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    wsb = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)
    print('done.\n')
    print('Dynesty fitting for fiber A\n')
    aberration_corrections.chromatic_fit(wsa_data, wsa, 'a')
    print('Dynesty fitting for fiber B\n')
    aberration_corrections.chromatic_fit(wsb_data, wsb, 'b')
    print('Chromatic aberrations coefficients saved.')


def fit_chromatic_aberrations_date(date, kin, fib):
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    print('Calculating chromatic aberration coefficients')
    print('Loading CARMENES data...', date),
    data = ws_load.read_ws(date, kin, fib)
    model = read_instrument_model(date, kin, fib)
    residuals = data['posm'].values - model['x'].values
    #plt.plot(data['posm'].values, residuals, 'k.')
    #plt.show()
    #plt.clf()
    print('done.\n')
    print('Dynesty fitting for fiber '+fib+'\n')
    model = aberration_corrections.chromatic_fit_date(date, data, model, fiber)
    #residuals = data['posm'].values - model['x'].values
    #print('Dynesty fitting for fiber B\n')
    #aberration_corrections.chromatic_fit(wsb_data, wsb, 'b')
    #print('Chromatic aberrations coefficients saved.')


def fit_optical_aberrations():
    print('Calculating optical aberration coefficients')
    print('Loading CARMENES data...'),
    wsa_data, wsb_data = ws_load.carmenes_vis_ws()
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    print('done.')
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state_a = parameters.load_sa('a')
    init_state_b = parameters.load_sa('b')
    print('MOES parameters loaded.')
    temps = env_data.get_temps()
    pressure = env_data.get_p()
    init_state_a[-1] = pressure
    init_state_b[-1] = pressure
    print('Tracing rays... '),
    wsa = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    wsb = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)
    print('Done')
    print('Loading chromatic aberrations...'),
    wsa = aberration_corrections.chromatic_model_load(wsa, 'a')
    wsb = aberration_corrections.chromatic_model_load(wsb, 'b')
    print('done.')
    print('Calculating optical aberrations coefficients fiber A')
    aberration_corrections.optical_fit(wsa_data, wsa, 'a')
    print('Calculating optical aberrations coefficients fiber B')
    aberration_corrections.optical_fit(wsb_data, wsb, 'b')
    print('optical aberrations fit is done.')


def fit_optical_aberrations_date(date, kin, fib):
    print('Calculating optical aberration coefficients')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    print('Loading CARMENES data...'),
    data = ws_load.read_ws(date, kin, fib)
    print('Calculating chromatic aberration coefficients')
    model = read_instrument_model(date, kin, fib)
    print('Loading chromatic aberrations...')
    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model
    residuals = data['posm'].values - model['x'].values
    print('done.')
    #plt.plot(data['posm'].values, residuals, 'k.')
    #plt.show()
    #plt.clf()
    print('Calculating optical aberrations coefficients fiber ' + fib)
    model = aberration_corrections.optical_fit_date(date, data, model, fiber)
    print('Pre-fit residuals = ', np.sqrt(np.sum( residuals ** 2) / len(residuals)))
    residuals = data['posm'].values - model['x'].values
    print('Post-fit residuals = ', np.sqrt(np.sum(residuals ** 2) / len(residuals)))
    print('optical aberrations fit is done.')
    #plt.plot(data['posm'].values, residuals, 'k.')
    #plt.show()


def fit_tertiary_aberrations_date(date, kin, fib):
    print('Calculating optical aberration coefficients')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    print('Loading CARMENES data...'),
    data = ws_load.read_ws(date, kin, fib)
    print('Calculating chromatic aberration coefficients')
    model = read_instrument_model(date, kin, fib)
    print('Loading chromatic aberrations...')
    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model
    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model

    model = poly_fit_date(data, model, 4)
    print('done.')
    residuals = data['posm'].values - model['x'].values
    plt.plot(model['wave']*model['order'], residuals, 'k.')
    plt.show()
    plt.clf()
    print('Fitting tertiary aberrations...')
    model = aberration_corrections.tertiary_fit_date(date, data, model, fiber)
    #plt.plot(data['posm'].values, residuals, 'k.')
    #plt.show()


def plot_ws_nopoly(date, kin, fib):
    print('Calculating optical aberration coefficients')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    print('Loading CARMENES data...', date),
    data = ws_load.read_ws(date, kin, fib)
    print('Calculating chromatic aberration coefficients')
    model = read_instrument_model(date, kin, fib)
    print('Loading chromatic aberrations...')
    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model
    print('done.')
    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model

    data['posm_norm'] = ((data['posm'].values - 4250. / 2) / (4250 / 2))
    model['x_norm'] = ((model['x'].values - 4250. / 2) / (4250 / 2))
    residuals = data['posm'].values - model['x'].values
    rms = np.sqrt(np.sum(residuals ** 2) / len(residuals))
    print('residuals WS moes = ', rms)
    residuals_data = data['posm'].values - data['posc'].values
    rms2 = np.sqrt(np.sum(residuals_data ** 2) / len(residuals_data))
    print('residuals WS caracal = ', rms2)
    #plt.plot(data['posm_norm'], residuals, 'k.', zorder=20)
    #plt.plot(data['posm_norm'], residuals_data, 'r.', zorder=0, alpha=0.5)
    #plt.show()
    plt.clf()

    # calculate new x's and y's
    x_new = np.linspace(0., 4250, 4250)
    y_new = f(x_new)
    xaux = np.linspace(0,4250,1)
    print(xaux)
    plt.plot(data['posm'].values, residuals_data, 'k.', zorder=0, alpha=0.5)
    plt.plot(data['posm'].values, residuals, 'r.', zorder=20, alpha=0.5)
    #plt.plot(x_new, y_new, 'ro', zorder=20)
    plt.show()


def poly_fit_date(data, model, deg):
    residuals = data['posm'].values - model['x'].values
    z = np.polyfit(data['posm'], residuals, deg)
    f = np.poly1d(z)

    polymodel = f(data['posm'].values)
    model['x'] = model['x'] + polymodel
    residuals = data['posm'].values - model['x'].values
    rms = np.sqrt(np.sum(residuals ** 2) / len(residuals))
    #print('residuals WS moes post poly = ', rms)
    return model#, rms


def sinefunc_fit_date(data, model, P):
    residuals = data['posm'].values - model['x'].values
    Pmax = 700.
    Pmin = 200.
    poptc, pcovc = curve_fit(sine, data['posm'], residuals,
                             p0=[max(residuals), 1.,
                                 0., 0.])
    xrange = np.linspace(0, 4096)
    #print(xrange)
    residuals = data['posm'].values - model['x'].values
    plt.plot(xrange, sine(xrange, *poptc), 'b-', zorder=20)
    plt.plot(data['posm'].values, residuals, 'k.', zorder=0, alpha=0.3)
    plt.show()
    plt.clf()

    sinemodel = sine(data['posm'].values, *poptc)
    model['x'] = model['x'] + sinemodel
    residuals = data['posm'].values - model['x'].values
    plt.plot(xrange, sine(xrange, *poptc), 'b-', zorder=20)
    plt.plot(data['posm'].values, residuals, 'k.', zorder=0, alpha=0.3)
    plt.show()
    plt.clf()




    #z = np.polyfit(data['posm'], residuals, deg)
    #f = np.poly1d(z)

    #polymodel = f(data['posm'].values)
    #model['x'] = model['x'] + polymodel
    #residuals = data['posm'].values - model['x'].values
    #rms = np.sqrt(np.sum(residuals ** 2) / len(residuals))
    #print('residuals WS moes post poly = ', rms)
    return model#, rms


def fit_joint_fit_date(date, kin, fib):
    print('Calculating optical aberration coefficients')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    print('Loading CARMENES data...'),
    data = ws_load.read_ws(date, kin, fib)
    print('Calculating chromatic aberration coefficients')
    model = read_instrument_model(date, kin, fib)
    aberration_corrections.joint_fit_date(date, data, model, fiber)


def fit_poly_fit_date(date, kin, fib):
    print('Calculating optical aberration coefficients')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    print('Loading CARMENES data...'),
    data = ws_load.read_ws(date, kin, fib)
    model = read_instrument_model(date, kin, fib)

    print('Loading chromatic aberrations...')
    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model
    print('done.')
    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model

    residuals_model = data['posm'].values - model['x'].values
    residuals_data = data['posm'].values - data['posc'].values
    rms = np.sqrt(np.sum(residuals_data ** 2) / len(residuals_data))
    rms2 = np.sqrt(np.sum(residuals_model ** 2) / len(residuals_model))
    print(rms, rms2)
    plt.plot(data['posm'].values, residuals_data, 'k.', alpha=0.5, zorder=0)
    plt.plot(data['posm'].values, residuals_model, 'r.', alpha=0.8, zorder=20)
    plt.show()
    plt.clf()
    aberration_corrections.poly_fit_date(date, data, model, fiber)


def read_instrument_model(date, kin, fib):
    data = ws_load.read_ws(date, kin, fib)
    spec = ws_load.spectrum_from_data(data)

    #init_state = parameters.load_date(fib, date)
    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model = vis_spectrometer.tracing(spec, init_state, fib, temps)
    return model


def read_full_model(date, kin, fib):
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    #data = ws_load.read_ws(date, kin, fib)
    data = ws_load.load_ws(date, kin, fib)
    #print(data)
    spec = ws_load.spectrum_from_data(data)

    #init_state = parameters.load_date(fib, date)
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

    polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date + '/'
    poly_coefs = pd.read_csv(polydir + 'poly_coefs_' + fiber + '.csv', sep=',')
    z = [poly_coefs['p0'].values[0],
         poly_coefs['p1'].values[0],
         poly_coefs['p2'].values[0],
         poly_coefs['p3'].values[0],
         poly_coefs['p4'].values[0]]
    f = np.poly1d(z)
    polymodel = f(data['posm'].values)
    model['x'] = model['x'] + polymodel
    model['wll'] = data['wll'].values
    #residuals = data['posm'].values - model['x'].values
    #residuals_data = data['posm'].values - data['posc'].values
    #plt.plot(data['posm'].values, residuals_data, 'k.', alpha=0.5, zorder=0)
    #plt.plot(data['posm'].values, residuals, 'r.', alpha=0.1, zorder=20)
    #plt.title(date)
    #plt.savefig('plots/ws_b/ws_'+date+'.png')
    #plt.show()
    #from PyAstronomy.pyTiming import pyPeriod
    #plt.clf()
    #plt.close()

    '''
    plt.figure(figsize=[8, 4])
    fapLevels = np.array([0.1, 0.05, 0.01])
    # Obtain the associated power thresholds
    clp = pyPeriod.Gls((data['posm'].values, residuals), norm="ZK", fbeg=1.e-3, fend=1.)
    plevels = clp.powerLevel(fapLevels)
    print(plevels)
    for i in range(len(fapLevels)):
        plt.plot([min(1./clp.freq), max(1./clp.freq)], [plevels[i]] * 2, 'r--',
                 label="FAP = %4.1f%%" % (fapLevels[i] * 100))

    plt.plot(1/clp.freq, clp.power, 'b.-')
    #plt.show()
    plt.clf()
    P = max(1/clp.freq)
    sinefunc_fit_date(data, model, P)
    '''
    return model


def read_full_model_spec(date, kin, fib):
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    spec = echelle_orders.initialize()

    init_state = parameters.load_date(fib, date)
    temps = env_data.get_T_at_ws(date)
    pressure = env_data.get_P_at_ws(date)
    init_state[-1] = pressure
    model = vis_spectrometer.tracing(spec, init_state, fib, temps)
    model = model.loc[model['x'] >= 0]
    model = model.loc[model['x'] <= 4250]
    model = model.loc[model['y'] >= 0]
    model = model.loc[model['y'] <= 4200]

    chromatic_coeffs = chromatic_aberrations.load_coeffs(date, fiber)
    chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
    model['x'] = model['x'] + chromatic_model

    optical_coeffs = optical_aberrations.load_coeffs(date, fiber)
    optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
    model['x'] = model['x'] + optical_model

    polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date + '/'
    poly_coefs = pd.read_csv(polydir + 'poly_coefs_' + fiber + '.csv', sep=',')
    z = [poly_coefs['p0'].values[0],
         poly_coefs['p1'].values[0],
         poly_coefs['p2'].values[0],
         poly_coefs['p3'].values[0],
         poly_coefs['p4'].values[0]]
    f = np.poly1d(z)
    polymodel = f(model['x'].values)
    model['x'] = model['x'] + polymodel

    #residuals = data['posm'].values - model['x'].values
    #residuals_data = data['posm'].values - data['posc'].values
    #plt.plot(data['posm'].values, residuals_data, 'k.', alpha=0.5, zorder=0)
    #plt.plot(model['x'].values, model['y'], 'r.', alpha=0.1, zorder=20)
    #plt.show()
    #plt.title(date)
    #plt.savefig('plots/ws_b/ws_'+date+'.png')
    #plt.show()
    #from PyAstronomy.pyTiming import pyPeriod
    #plt.clf()
    #plt.close()

    '''
    plt.figure(figsize=[8, 4])
    fapLevels = np.array([0.1, 0.05, 0.01])
    # Obtain the associated power thresholds
    clp = pyPeriod.Gls((data['posm'].values, residuals), norm="ZK", fbeg=1.e-3, fend=1.)
    plevels = clp.powerLevel(fapLevels)
    print(plevels)
    for i in range(len(fapLevels)):
        plt.plot([min(1./clp.freq), max(1./clp.freq)], [plevels[i]] * 2, 'r--',
                 label="FAP = %4.1f%%" % (fapLevels[i] * 100))

    plt.plot(1/clp.freq, clp.power, 'b.-')
    #plt.show()
    plt.clf()
    P = max(1/clp.freq)
    sinefunc_fit_date(data, model, P)
    '''
    return model


def read_full_model_from_data(date, kin, fib, data):
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'

    #print(data)
    spec = ws_load.spectrum_from_data(data)
    #init_state = parameters.load_date(fib, date)
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

    polydir = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + date + '/'
    poly_coefs = pd.read_csv(polydir + 'poly_coefs_' + fiber + '.csv', sep=',')
    z = [poly_coefs['p0'].values[0],
         poly_coefs['p1'].values[0],
         poly_coefs['p2'].values[0],
         poly_coefs['p3'].values[0],
         poly_coefs['p4'].values[0]]
    f = np.poly1d(z)
    polymodel = f(data['posm'].values)
    model['x'] = model['x'] + polymodel

    #residuals = data['posm'].values - model['x'].values
    #residuals_data = data['posm'].values - data['posc'].values
    #plt.plot(data['posm'].values, residuals_data, 'k.', alpha=0.5, zorder=0)
    #plt.plot(data['posm'].values, residuals, 'r.', alpha=0.1, zorder=20)
    #plt.title(date)
    #plt.savefig('plots/ws_b/ws_'+date+'.png')
    #plt.show()
    #from PyAstronomy.pyTiming import pyPeriod
    #plt.clf()
    #plt.close()

    '''
    plt.figure(figsize=[8, 4])
    fapLevels = np.array([0.1, 0.05, 0.01])
    # Obtain the associated power thresholds
    clp = pyPeriod.Gls((data['posm'].values, residuals), norm="ZK", fbeg=1.e-3, fend=1.)
    plevels = clp.powerLevel(fapLevels)
    print(plevels)
    for i in range(len(fapLevels)):
        plt.plot([min(1./clp.freq), max(1./clp.freq)], [plevels[i]] * 2, 'r--',
                 label="FAP = %4.1f%%" % (fapLevels[i] * 100))

    plt.plot(1/clp.freq, clp.power, 'b.-')
    #plt.show()
    plt.clf()
    P = max(1/clp.freq)
    sinefunc_fit_date(data, model, P)
    '''
    return model


def calculate_residuals(date, kin, fib):
    data = ws_load.read_ws(date, kin, fib)
    model = read_instrument_model(date, kin, fib)
    N = len(model)
    rms_x = np.sqrt(np.sum((model['x'].values - data['posm'].values) ** 2) / N)
    residuals_x = model['x'].values - data['posm'].values
    plt.figure(figsize=[14, 6])
    plt.scatter(data['posm'], residuals_x, marker='.',c=model['wave'].values ,alpha=0.8, cmap=cm.viridis)
    clb = plt.colorbar(orientation='vertical')
    clb.ax.set_ylabel(r'Wavelength ($\mu m$)', size=12)
    plt.xlabel('x (pix)')
    plt.ylabel(r'x$_{model}$ - x$_{data}$ (pix)')
    plt.title('CARMENES VIS, ' + date)
    plt.tight_layout()
    #plt.show()
    plt.close()
    print('Residuals in x = ', rms_x, ' pix')


    #print(model)
    model = load_aberrations(model, 'A')
    residuals_x = model['x'].values - data['posm'].values
    plt.figure(figsize=[14, 6])
    plt.scatter(data['posm'], residuals_x, marker='.', c=model['wave'].values, alpha=0.8, cmap=cm.viridis)
    clb = plt.colorbar(orientation='vertical')
    clb.ax.set_ylabel(r'Wavelength ($\mu m$)', size=12)
    plt.xlabel('x (pix)')
    plt.ylabel(r'x$_{model}$ - x$_{data}$ (pix)')
    plt.title('CARMENES VIS, ' + date)
    plt.tight_layout()
    plt.show()
    plt.close()
    print('Residuals in x = ', rms_x, ' pix')


def fit_all_chromatic(fib, kin):
    path_data = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/'
    datesdirs = glob.glob(path_data + '*')
    #print(dates)
    dates, jd = [], []
    for date in datesdirs:
        d = date[len(path_data):]
        disot = Time(d+'T12:00:00.0', format='isot')
        dates.append(d)
        jd.append(disot.jd)

    daux = pd.DataFrame()
    daux['dates'] = dates
    daux['jd'] = jd
    daux = daux.sort_values(by='jd')
    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])

    for d in dout:
        print('Chromatic date fit for ', d)
        fit_chromatic_aberrations_date(d, kin, fib)


def fit_all_optical(fib, kin):
    path_data = 'data/aberrations_coefficients/optical_coefficients_timeseries/'
    datesdirs = glob.glob(path_data + '*')
    #print(dates)
    dates, jd = [], []
    for date in datesdirs:
        d = date[len(path_data):]
        disot = Time(d+'T12:00:00.0', format='isot')
        dates.append(d)
        jd.append(disot.jd)

    daux = pd.DataFrame()
    daux['dates'] = dates
    daux['jd'] = jd
    daux = daux.sort_values(by='jd')
    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])

    for d in dout:
        print('Chromatic date fit for ', d)
        fit_chromatic_aberrations_date(d, kin, fib)
        print('Optical date fit for ', d)
        fit_optical_aberrations_date(d, kin, fib)


def fit_all_full(fib):
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

    jdini = Time('2019-05-11T12:00:00.0', format='isot')
    jdini = jdini.jd
    daux = daux.loc[daux['jd'] >= jdini]

    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])
    #print(dout)
    kin = 'hcl'
    for d in dout:
        print('MOES model')
        simulated_annealing_fit_date(d, fib)
        print('Chromatic date fit for ', d)
        fit_chromatic_aberrations_date(d, kin, fib)
        print('Optical date fit for ', d)
        fit_optical_aberrations_date(d, kin, fib)


def load_joint_fit(date, kin, fib):
    print('Calculating optical aberration coefficients')
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
    basedir = '/home/marcelo/codes/moes/carmenes/data/aberrations_coefficients/joint_coefficients_timeseries/' + date + '/'
    print('Loading CARMENES data...'),
    data = ws_load.read_ws(date, kin, fib)
    print('Calculating chromatic aberration coefficients')
    model = read_instrument_model(date, kin, fib)
    joint_coeffs = pd.read_pickle(basedir + 'best_fit_pars_' + fiber + '.pkl')
    #print(joint_coeffs)
    #a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = joint_coeffs[0], joint_coeffs[1], joint_coeffs[2], joint_coeffs[3], joint_coeffs[4], joint_coeffs[5], joint_coeffs[6], joint_coeffs[7], joint_coeffs[8], joint_coeffs[9], joint_coeffs[10], joint_coeffs[11], joint_coeffs[12]
    coeffs = [joint_coeffs['c0'], joint_coeffs['c1'], joint_coeffs['c2'], joint_coeffs['c3'], joint_coeffs['c4'], joint_coeffs['c5'], joint_coeffs['c6'], joint_coeffs['c7'], joint_coeffs['c8'], joint_coeffs['c9'], joint_coeffs['c10'], joint_coeffs['c11'], joint_coeffs['c12']]
    jointmodel = joint_aberrations.function(coeffs, model)
    model['x'] = model['x'] + jointmodel

    residuals = data['posm'].values - model['x'].values
    rms = np.sqrt(np.sum(residuals ** 2) / len(residuals))
    print(rms)
    plt.plot(data['posm'].values, residuals, 'k.')
    plt.show()


def plot_all_dates(fib):
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

    jdini = Time('2018-01-30T12:00:00.0', format='isot')
    jdini = jdini.jd
    daux = daux.loc[daux['jd'] >= jdini]

    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])
    # print(dout)
    kin = 'hcl'

    for d in dates:
        read_full_model(d, kin, fib)


def fit_all_poly(fib, deg):
    if fib == 'A':
        fiber = 'a'
    elif fib == 'B':
        fiber = 'b'
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

    #jdini = Time('2019-05-11T12:00:00.0', format='isot')
    #jdini = jdini.jd
    #daux = daux.loc[daux['jd'] >= jdini]

    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])
    # print(dout)
    kin = 'hcl'
    outdir = 'data/aberrations_coefficients/poly_coefficients_timeseries/'
    for d in dout:
        print(d),
        ddir = outdir + d + '/'
        if not os.path.exists(ddir):
            os.mkdir(ddir)
        data = ws_load.read_ws(d, kin, fib)
        # print(data)
        spec = ws_load.spectrum_from_data(data)

        # init_state = parameters.load_date(fib, date)
        init_state = parameters.load_date(fib, d)
        temps = env_data.get_T_at_ws(d)
        pressure = env_data.get_P_at_ws(d)
        init_state[-1] = pressure
        model = vis_spectrometer.tracing(spec, init_state, fib, temps)

        chromatic_coeffs = chromatic_aberrations.load_coeffs(d, fiber)
        chromatic_model = chromatic_aberrations.function(np.array(model['wave'].values), chromatic_coeffs)
        model['x'] = model['x'] + chromatic_model

        optical_coeffs = optical_aberrations.load_coeffs(d, fiber)
        optical_model = optical_aberrations.function_seidel(model, optical_coeffs)
        model['x'] = model['x'] + optical_model

        #model = poly_fit_date(data, model, 4)
        residuals = data['posm'].values - model['x'].values
        z = np.polyfit(data['posm'], residuals, deg)
        f = np.poly1d(z)
        print(z[0], z[1], z[2], z[3], z[4])
        fout = open(ddir + 'poly_coefs_'+fiber+'.csv', 'w')
        fout.write('p0,p1,p2,p3,p4\n')
        fout.write('%e,%e,%e,%e,%e' %(z[0].astype(float),
                                      z[1].astype(float),
                                      z[2].astype(float),
                                      z[3].astype(float),
                                      z[4].astype(float)))
        #dfout = pd.DataFrame()
        #dfout['p0'] =
        #dfout['p1'] = z[1].astype(float)
        #dfout['p2'] = z[2].astype(float)
        #dfout['p3'] = z[3].astype(float)
        #dfout['p4'] = z[4].astype(float)
        #print(dfout)
        #dfout.to_csv()
        #print('File written.')
        #polymodel = f(data['posm'].values)
        #model['x'] = model['x'] + polymodel
        #residuals = data['posm'].values - model['x'].values
        #rms = np.sqrt(np.sum(residuals ** 2) / len(residuals))
        #print(rms)
        #plt.plot(data['posm'].values, residuals, 'k.')
        #plt.show()
        #plt.clf()
        #plt.close()


def prepare_lineset():
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

    # jdini = Time('2019-05-11T12:00:00.0', format='isot')
    # jdini = jdini.jd
    # daux = daux.loc[daux['jd'] >= jdini]

    dout = []
    for k in range(len(daux)):
        dout.append(daux['dates'].values[k])
    # print(dout)
    kin = 'hcl'
    #outdir = 'data/aberrations_coefficients/poly_coefficients_timeseries/'

    # Initial spectrum
    dataA = ws_load.read_ws(dout[0], kin, 'A')
    dataA['order'] = dataA['order'].astype(int)
    dataA['wll'] = dataA['wll'].round(2)

    dataB = ws_load.read_ws(dout[0], kin, 'B')
    dataB['order'] = dataB['order'].astype(int)
    dataB['wll'] = dataB['wll'].round(2)

    dataAB = pd.merge(dataA, dataB, how='inner', on=['wll', 'order'], suffixes=['_a', '_b'])

    spectrum = pd.DataFrame()
    spectrum['wll'] = dataAB['wll'].values
    spectrum['order'] = dataAB['order'].values
    spectrum = spectrum.drop_duplicates(subset=['wll', 'order'])

    for d in dout:
        dataA = ws_load.read_ws(d, kin, 'A')
        dataA['order'] = dataA['order'].astype(int)
        dataA['wll'] = dataA['wll'].round(2)

        dataB = ws_load.read_ws(d, kin, 'B')
        dataB['order'] = dataB['order'].astype(int)
        dataB['wll'] = dataB['wll'].round(2)

        dataAB = pd.merge(dataA, dataB, how='inner', on=['wll', 'order'], suffixes=['_a', '_b'])
        specdate = pd.DataFrame()
        specdate['wll'] = dataAB['wll'].round(2).values
        specdate['order'] = dataAB['order'].values.astype(int)
        specdate = specdate.drop_duplicates(subset=['wll', 'order'])

        specaux = pd.merge(specdate, spectrum, how='inner', on=['wll', 'order'])
        specaux = specaux.drop_duplicates(subset=['wll', 'order'])

        spectrum = specaux
        print(d, len(spectrum))

        #dataA = dataA.round(ndigits)
        #dataB = dataB.round(ndigits)
        #dataAB = pd.merge(dataA, dataB, how='inner', on=['wll', 'order'], suffixes=['_a', '_b'])
        #print(dataAB['wll'], dataAB['order'])
        #waves = np.unique(dataAB['wll'].values)
        #datawaves =
        #spectrumAB = pd.DataFrame()

        #spectrum_aux = pd.merge(dataAB, spectrum, how='inner', on=['wll', 'order'])
        #print(spectrum_aux)
        #waves = np.unique(spectrum_aux['wll'].values)
    out = pd.DataFrame()
    out['order'] = spectrum['order'].values
    out['wll'] = spectrum['wll'].values
    out.to_csv('spectrum.csv', index=False)



if __name__ == '__main__':
    
    fib = 'B'
    date = '2017-10-21'
    kin = 'hcl'
    #fit_all_poly('A', 4)
    #fit_all_poly('B', 4)
    #fit_all_full(fib)
    model = read_full_model(date, kin, fib)
    print(model['wll'])
    #prepare_lineset()
    #plot_all_dates('B')

    # date = '2017-10-20'
    # kin = 'hcl'
    #calculate_residuals(date, kin, fib)
    #simulated_annealing_fit_date(date, fib)
    #fit_chromatic_aberrations_date(date, kin, fib)
    #fit_all_chromatic(fib)
    #fit_all_optical(fib)
    #fit_joint_fit_date(date, kin, fib)
    #load_joint_fit(date, kin, fib)
    #plot_ws_nopoly(date, kin, fib)


