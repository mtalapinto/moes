import time
import numpy as np
from simanneal import Annealer
from optics import vis_spectrometer
from optics import parameters
import pandas as pd
import matplotlib.pyplot as plt
from env_data import env_data
import ws_load


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
        delta_0 = 1e-8
        delta_1 = 1e-8
        delta_2 = 1e-7 #1e-4
        delta_3 = 1e-7 #1e-2

        if self.fib == 'a':
            self.state = [np.random.normal(self.state[0], 1e-5),  # slit_dec_x_a
                          np.random.normal(self.state[1], 1e-5),  # slit_dec_y_a
                          np.random.normal(self.state[2], 1e-99),  # slit_dec_x_b
                          np.random.normal(self.state[3], 1e-99),  # slit_dec_y_b
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
                          np.random.normal(self.state[30], 5*delta_all),  # d_cam_ff
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
        else:
            self.state = [np.random.normal(self.state[0], 1e-99),  # slit_dec_x_a
                          np.random.normal(self.state[1], 1e-99),  # slit_dec_y_a
                          np.random.normal(self.state[2], 1e-5),  # slit_dec_x_b
                          np.random.normal(self.state[3], 1e-5),  # slit_dec_y_b
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
                          np.random.normal(self.state[30], 5*delta_all),  # d_cam_ff
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
        
        chi2x = np.sum((self.ws_data[:, 3] - ws_mod[:, 2])**2 / self.ws_data[:,4])
        chi2y = np.sum((self.ws_data[:, 5] - ws_mod[:, 3])**2 / self.ws_data[:,4])
        rchi2x = chi2x/len(self.state)
        rchi2y = chi2y/len(self.state)
        rchi2 = np.sqrt(chi2x ** 2 + chi2y ** 2)
        e = rchi2x
        return e


def moes_carmenes_vis(date):
    # We load the data
    #date = '2017-08-01'
    print('Moe\'s Simulated Annealing fit - CARMENES VIS -', date)
    wsa_data, wsb_data = ws_load.carmenes_vis_ws(date)
    wsa_data = np.array(wsa_data)
    wsb_data = np.array(wsb_data)
    spec_a = ws_load.spectrum_from_ws(wsa_data)
    spec_b = ws_load.spectrum_from_ws(wsb_data)
    init_state_a = parameters.load_sa('a')
    init_state_b = parameters.load_sa('b')
    temps = env_data.get_temps_date(date)
    wsa_model = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    wsb_model = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)
    chi2a = np.sum((wsa_data[:, 3] - wsa_model[:, 2])**2/wsa_data[:,4])
    rchi2a = chi2a/len(init_state_a)
    chi2b = np.sum((wsb_data[:, 3] - wsb_model[:, 2])**2/wsb_data[:,4])
    rchi2b = chi2b/len(init_state_b)
    
    plt.plot(wsa_model[:,2], wsa_data[:, 3] - wsa_model[:, 2],'b.')
    plt.show()
    res_x_a = np.sqrt(np.mean((wsa_data[:, 3] - wsa_model[:, 2])**2))
    res_y_a = np.sqrt(np.mean((wsa_data[:, 5] - wsa_model[:, 3])**2))
    res_x_b = np.sqrt(np.mean((wsb_data[:, 3] - wsb_model[:, 2])**2))
    res_y_b = np.sqrt(np.mean((wsb_data[:, 5] - wsb_model[:, 3])**2))
    
    print('Residuals')
    print('Fiber A')
    print('res_x =', res_x_a, ', res_y = ', res_y_a)
    print('Fiber B')
    print('res_x =', res_x_b, ', res_y = ', res_y_b)
    
    i = 0
    n = 0
    print('Starting fit...')
    while i < n:
        # initial state, initial set of parameters
        # Fiber A
        # Initial state
        print('Iteration no. ', i)
        print('Fiber A')
        init_state_a = parameters.load_sa('a')
        write_old_a = parameters.write_old(init_state_a)
        temps = env_data.get_temps_date(date)

        # Simulated annealing routine for fiber A
        wsa_fit = WavelengthSolutionFit(init_state_a, wsa_data, spec_a, temps, 'a')
        wsa_fit.steps = 1000
        wsa_fit.Tmax = 5000
        wsa_fit.Tmin = 1e-10

        state_a, e_a = wsa_fit.anneal()
        parameters.write_sim(state_a, 'a')
        print('\n')
        # Fiber B
        print('Fiber B')
        init_state_b = parameters.load_sa('b')
        write_old_b = parameters.write_old(init_state_b)
        temps = env_data.get_temps_date(date)

        # Simulated annealing routine for fiber B
        wsb_fit = WavelengthSolutionFit(init_state_b, wsb_data, spec_b, temps, 'b')
        wsb_fit.steps = 1000
        wsb_fit.Tmax = 25000
        wsb_fit.Tmin = 1e-10

        state_b, e_b = wsb_fit.anneal()
        parameters.write_sim(state_b, 'b')
        print('\n')
        i += 1
    
    
    init_state_a = parameters.load_sa('a')
    init_state_b = parameters.load_sa('b')
    temps = env_data.get_temps_date(date)
    wsa_model = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    wsb_model = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)
    
    res_x_a = np.sqrt(np.mean((wsa_data[:, 3] - wsa_model[:, 2])**2))
    res_y_a = np.sqrt(np.mean((wsa_data[:, 5] - wsa_model[:, 3])**2))
    res_x_b = np.sqrt(np.mean((wsb_data[:, 3] - wsb_model[:, 2])**2))
    res_y_b = np.sqrt(np.mean((wsb_data[:, 5] - wsb_model[:, 3])**2))
    
    print('Residuals after fit')
    print('Fiber A')
    print('res_x =', res_x_a, ', res_y = ', res_y_a)
    print('Fiber B')
    print('res_x =', res_x_b, ', res_y = ', res_y_b)
    

if __name__ == '__main__':
    
    date = '2017-08-01'
    moes_carmenes_vis(date)
