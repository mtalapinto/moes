import ws_load
from optics import env_data
from optics import parameters
from optics import vis_spectrometer
import numpy as np
import chromatic_aberrations
import optical_aberrations
import matplotlib.pyplot as plt
import pandas as pd
from optics import CCD_vis
from astropy.time import Time
from optics import camera
from optics import echelle_orders
import os
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import os.path
#from astropy.timeseries import LombScargle
import scipy.signal as signal
import warnings
warnings.filterwarnings("ignore")


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta


def chromatic_fit(ws_data, ws_model, fiber):
    path_chromatic = 'data/aberrations_coefficients/chromatic/'
    if not os.path.exists(path_chromatic):
        os.mkdir(path_chromatic)

    file_chromatic_coeffs = open(path_chromatic+'chrome_coeffs_'+str(fiber)+'.dat', 'w')
    file_chromatic_coeffs.write('coor,a0,a1,a2,a3\n')
    
    chromatic_coeffs_x = chromatic_aberrations.correct_dyn(ws_data, ws_model, 'x', fiber)
    chromatic_model_x = chromatic_aberrations.function(np.array(ws_model[:, 1]).astype(np.float), chromatic_coeffs_x)
    
    #plt.plot(ws_model[:, 1], ws_data[:, 3] - ws_model[:, 2], 'k+')
    #plt.plot(ws_model[:, 1], chromatic_model_x, 'r+')
    #plt.show()
    #plt.clf()
    
    file_chromatic_coeffs.write('x,%f,%f,%f,%f\n' %(chromatic_coeffs_x[0], chromatic_coeffs_x[1],
                                                    chromatic_coeffs_x[2], chromatic_coeffs_x[3]))
    ws_model[:, 2] = ws_model[:, 2] + chromatic_model_x
    
    #plt.plot(ws_model[:, 1], ws_data[:, 3] - ws_model[:, 2], 'b+')
    #plt.show()
    #plt.clf()
    return ws_model


def optical_fit(ws_data, ws_model, fiber):
    path_optical = 'data/aberrations_coefficients/optical/'
    if not os.path.exists(path_optical):
        os.mkdir(path_optical)
    file_out = open(path_optical + 'seidel_coefs_' + str(fiber) + '.dat', 'w')
    file_out.write('coord,a0,a1,a2,a3,a4,a5,a6,a7,a8\n')

    # Correction in X
    seidel_coeffs_x = optical_aberrations.correct_seidel_dyn(ws_data, ws_model, 'x', fiber)
    file_out.write('x,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (seidel_coeffs_x[0],
                                                       seidel_coeffs_x[1],
                                                       seidel_coeffs_x[2],
                                                       seidel_coeffs_x[3],
                                                       seidel_coeffs_x[4],
                                                       seidel_coeffs_x[5],
                                                       seidel_coeffs_x[6],
                                                       seidel_coeffs_x[7],
                                                       seidel_coeffs_x[8]
                                                       ))

    ws_model_norm = CCD_vis.pix2mm_model(ws_model, 'x')
    rho, theta = cart2pol(ws_model_norm[:, 2].astype(np.float), ws_model_norm[:, 3].astype(np.float))
    #plt.clf()
    #plt.plot(ws_data[:, 3], ws_data[:, 3] - ws_model[:, 2], 'k+')
    seidel_aberrations_model_x = optical_aberrations.function_seidel_aberrations(seidel_coeffs_x, rho, theta)
    # plt.plot(ws_data[:, 3], seidel_aberrations_model_x, 'r+')
    # plt.show()
    # plt.clf()
    ws_model[:, 2] = ws_model[:, 2] + seidel_aberrations_model_x
    # plt.plot(ws_data[:, 3], ws_data[:, 3] - ws_model[:, 2], 'b+')
    #plt.show()
    #plt.clf()
    return ws_model


def chromatic_model_load(ws_model, fiber):
    path_chromatic = 'data/aberrations_coefficients/chromatic/'
    coefsx = pd.read_csv(path_chromatic + 'chrome_coeffs_' + str(fiber) + '.dat', sep=',')
    a0x = coefsx['a0'].values[0]
    a1x = coefsx['a1'].values[0]
    a2x = coefsx['a2'].values[0]
    a3x = coefsx['a3'].values[0]
    chromatic_coeffs_x = [a0x, a1x, a2x, a3x]
    chromatic_model_x = chromatic_aberrations.function(np.array(ws_model[:, 1]).astype(np.float), chromatic_coeffs_x)
    ws_model[:, 2] = ws_model[:, 2] + chromatic_model_x
    return ws_model


def optical_model_load(ws_model, fiber):
    path_optical = 'data/aberrations_coefficients/optical/'

    coefs = pd.read_csv(path_optical + 'seidel_aberrations_coefs_' + str(fiber) + '.dat', sep=',')

    a0 = coefs['a0'].values[0]
    a1 = coefs['a1'].values[0]
    a2 = coefs['a2'].values[0]
    a3 = coefs['a3'].values[0]
    a4 = coefs['a4'].values[0]
    a5 = coefs['a5'].values[0]
    a6 = coefs['a6'].values[0]
    a7 = coefs['a7'].values[0]
    a8 = coefs['a8'].values[0]

    coefs_x = [a0, a1, a2, a3, a4, a5, a6, a7, a8]

    ws_model_norm = CCD_vis.pix2mm_model(ws_model, 'x')
    rho, theta = cart2pol(ws_model_norm[:, 2].astype(np.float), ws_model_norm[:, 3].astype(np.float))
    seidel_aberrations_x = optical_aberrations.function_seidel(coefs_x, rho, theta)
    ws_model[:, 2] = ws_model[:, 2] + seidel_aberrations_x

    return ws_model


if __name__ == '__main__':
    print('test')
