import warnings
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
import os
import utils
import dynesty


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
        chi2x = np.sqrt(np.mean((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2))
        #chi2x = np.sqrt(np.mean((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2))
        chi2y = np.sqrt(np.mean((self.ws_data[:, 5] - ws_mod[:, 3]) ** 2))
        #chi2y = np.sum((self.ws_data[:, 5] - ws_mod[:, 3]) ** 2 / self.ws_data[:, 4])
        #rchi2x = chi2x / len(self.state)
        #rchi2y = chi2y / len(self.state)
        rchi2 = np.sqrt(chi2x ** 2 + chi2y ** 2)
        e = rchi2
        #e = np.sqrt(np.mean((self.ws_data[:, 3] - ws_mod[:, 2]) ** 2))
        return e


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
    ws_model = aberration_corrections.chromatic_model_load(ws, fib)
    ws_model = aberration_corrections.optical_model_load(ws_model, fib)

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


def fit_instrument_model(style):
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


if __name__ == '__main__':
    
    #run_instrument_model()
    #fit_instrument_model('nested-sampling')
    #fit_chromatic_aberrations()
    fit_optical_aberrations()
