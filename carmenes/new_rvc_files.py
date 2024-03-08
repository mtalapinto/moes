import os
import glob
import pandas as pd
import numpy as np
from astropy.time import Time
from optics import env_data
import aberration_corrections
from optics import parameters
import ws_load
from optics import vis_spectrometer
import math
import chromatic_aberrations
import optical_aberrations
import plots
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaus(x, height, x0, sigma):
    return height * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))  # + offset


def get_sum(vec):
    fvec = np.sort(vec)
    fval = np.median(fvec)
    nn = int(np.around(len(fvec) * 0.15865))
    vali, valf = fval - fvec[nn], fvec[-nn] - fval
    return fval, vali, valf


def create_new_rvc_files():
    # Setting directories

    out_path_stars = 'data/stars/'
    if not os.path.exists(out_path_stars):
        os.mkdir(out_path_stars)

    #avc_data_path = '/home/marcelo/Documentos/moes/carmenes_v3.3/vis/data/CARM_VIS_AVC_201017/avcn/'
    #rvc_data_path = '/home/marcelo/Documentos/moes/carmenes/vis/data/CARM_VIS_RVC/'
    rvc_data_path = 'caracal_rvs/CARM_VIS_RVC/'
    rvcout_path = 'caracal_rvs/CARM_VIS_RVC_tzp/'
    if not os.path.exists(rvcout_path):
        os.mkdir(rvcout_path)

    tzpdata = pd.read_csv('data/all_stars_tzp_full.csv',sep=',')
    tzpmed_mean = np.median(tzpdata['tzp_med'].values)

    # Creating full outfile
    datebase = '2017-10-20T12:00:00'
    bjd_base = Time(datebase, format='isot').jd
    stars_path = glob.glob(rvc_data_path + '*/')
    #print(stars_path)
    print('Creating new RVC files corrected by the TZP...')
    nstars = len(stars_path)
    for i in range(len(stars_path)):
        star_id = stars_path[i][len(rvc_data_path):-1]
        rvc_data_per_star = pd.read_csv(rvc_data_path + str(star_id) + '/' + str(star_id) + '.rvc.dat', sep=' ',
                                        names=['bjd', 'rvc', 'e_rvc', 'drift', 'e_drift', 'rv', 'e_rv', 'berv',
                                               'sadrift'])
        rvc_data_per_star = rvc_data_per_star.dropna()
        rvc_data_per_star = rvc_data_per_star.loc[rvc_data_per_star['bjd'] > bjd_base]
        nobs = len(rvc_data_per_star)
        if nobs > 3:

            outpathstar = rvcout_path + str(star_id) + '/'
            if not os.path.exists(outpathstar):
                os.mkdir(outpathstar)
            fileout = open(outpathstar + str(star_id) + '.rvc.dat', 'w')
            print(star_id, ', ')
            print('star ', i, 'out of ', nstars)

            for k in range(nobs):
                print(star_id, 'observation no ', k, 'out of ', len(rvc_data_per_star))
                date_bjd = rvc_data_per_star['bjd'].values[k]
                rvc = rvc_data_per_star['rvc'].values[k]
                e_rvc = rvc_data_per_star['e_rvc'].values[k]
                drift = rvc_data_per_star['drift'].values[k]
                e_drift = rvc_data_per_star['e_drift'].values[k]
                rv = rvc_data_per_star['rv'].values[k]
                e_rv = rvc_data_per_star['e_rv'].values[k]
                berv = rvc_data_per_star['berv'].values[k]
                sadrift = rvc_data_per_star['sadrift'].values[k]

                dt = (int(date_bjd) + 0.5) - date_bjd
                if dt < 0:
                    date = Time(date_bjd, format='jd').isot[:10]
                    dateaux = Time(date + 'T12:00:00.0', format='isot').jd
                    dateaux = dateaux - 1
                    date_ws = Time(dateaux, format='jd').isot[:10]
                else:
                    date = Time(date_bjd, format='jd').isot[:10]
                    date_ws = date

                temps_bjd = env_data.get_temps_bjd(date_bjd)
                pressure = env_data.get_p_bjd(date_bjd)

                path_chromatic_a = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(
                    date_ws) + '/chrome_coeffs_a.dat'
                path_chromatic_b = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(
                    date_ws) + '/chrome_coeffs_b.dat'

                path_optical_a = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'seidel_coefs_a.dat'
                path_optical_b = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'seidel_coefs_b.dat'

                path_poly_a = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'poly_coefs_a.csv'
                path_poly_b = 'data/aberrations_coefficients/poly_coefficients_timeseries/' + str(
                    date_ws) + '/' + 'poly_coefs_b.csv'
                if os.path.exists(path_chromatic_a) and os.path.exists(path_chromatic_b) and os.path.exists(
                        path_optical_a) and os.path.exists(path_optical_b) and os.path.exists(
                    path_poly_a) and os.path.exists(path_poly_b):
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
                        spec = np.array([dd_data['order'].values, dd_data['wavec'].values.astype(float) * 1e-4]).T

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
                        rvscale_a_ws, rvscale_b_ws = plots.get_rvscale(spec, init_state_a, init_state_b, temps_ws,
                                                                 date_ws)

                        rvscale_a, rvscale_b = plots.get_rvscale(spec, init_state_a, init_state_b, temps_bjd, date_ws)

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
                                tzp_corr = tzp_m_median - tzpmed_mean
                                newrvc = rvc + tzp_corr
                                fileout.write('%f %f %f %f %f %f %f %f %f\n' %(date_bjd, newrvc, e_rvc, drift, e_drift, rv, e_rv, berv, sadrift))


            fileout.close()


def check_number_of_rvs_in_file():
    basedir = 'caracal_rvs/CARM_VIS_RVC_tzp/'
    stardirs = glob.glob(basedir + '*')
    print(stardirs)
    out = []
    for dir in stardirs:
        files = glob.glob(dir + '/*.dat')
        rvcs = pd.read_csv(files[0], delim_whitespace=True, names=['bjd', 'rvc', 'e_rvc', 'drift', 'e_drift', 'rv', 'e_rv', 'berv',
                                               'sadrift'])
        #print(rvcs)
        out.append(dir[len(basedir):])

    print(out)

if __name__ == '__main__':

    #create_new_rvc_files()
    check_number_of_rvs_in_file()