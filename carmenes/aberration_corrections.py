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
import joint_aberrations
import poly_fit


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


def chromatic_fit_date(date, ws_data, ws_model, fiber):
    path_chromatic = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(date) + '/'
    if not os.path.exists(path_chromatic):
        os.mkdir(path_chromatic)

    file_chromatic_coeffs = open(path_chromatic + 'chrome_coeffs_' + str(fiber) + '.dat', 'w')
    file_chromatic_coeffs.write('coor,a0,a1,a2,a3\n')

    chromatic_coeffs_x = chromatic_aberrations.correct_dyn(ws_data, ws_model, 'x', fiber, date)
    chromatic_model_x = chromatic_aberrations.function(np.array(ws_model['wave'].values), chromatic_coeffs_x)

    #plt.plot(ws_model['wave'].values, ws_data['posm'].values - ws_model['x'].values, 'k+')
    #plt.plot(ws_model['wave'].values, chromatic_model_x, 'r+')
    #plt.show()
    #plt.clf()

    file_chromatic_coeffs.write('x,%f,%f,%f,%f\n' % (chromatic_coeffs_x[0], chromatic_coeffs_x[1],
                                                     chromatic_coeffs_x[2], chromatic_coeffs_x[3]))
    ws_model['x'] = ws_model['x'] + chromatic_model_x
    residuals = ws_data['posm'].values - ws_model['x'].values
    print('Post chromatic fit residuals = ', np.sqrt(np.sum(residuals**2)/len(residuals)))
    #plt.plot(ws_model['wave'], residuals, 'b+')
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
    seidel_aberrations_model_x = optical_aberrations.function_seidel(seidel_coeffs_x, rho, theta)
    # plt.plot(ws_data[:, 3], seidel_aberrations_model_x, 'r+')
    # plt.show()
    # plt.clf()
    ws_model[:, 2] = ws_model[:, 2] + seidel_aberrations_model_x
    # plt.plot(ws_data[:, 3], ws_data[:, 3] - ws_model[:, 2], 'b+')
    #plt.show()
    #plt.clf()
    return ws_model


def optical_fit_date(date, ws_data, ws_model, fiber):
    path_optical = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + date + '/'
    if not os.path.exists(path_optical):
        os.mkdir(path_optical)
    file_out = open(path_optical + 'seidel_coefs_' + str(fiber) + '.dat', 'w')
    file_out.write('coord,a0,a1,a2,a3,a4,a5,a6,a7,a8\n')

    # Correction in X
    seidel_coeffs_x = optical_aberrations.correct_dyn(ws_data, ws_model, 'x', fiber, date)
    #seidel_coeffs_x = optical_aberrations.correct_seidel(ws_data, ws_model, 'x', fiber, date)
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
    file_out.close()

    xprime = (ws_model['x'].values - 4250. / 2) / (4250 / 2)
    yprime = (ws_model['y'].values - 4200. / 2) / (4200 / 2)

    rho = np.sqrt(xprime ** 2 + yprime ** 2)
    theta = np.arctan2(yprime, xprime)

    model = seidel_coeffs_x[0] * 4 * rho ** 3 + \
            seidel_coeffs_x[1] * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
            seidel_coeffs_x[2] * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
            seidel_coeffs_x[3] * 2 * rho + \
            seidel_coeffs_x[4] * (np.cos(theta) - rho * np.sin(theta)) + \
            seidel_coeffs_x[5] * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta)) + \
            seidel_coeffs_x[6] * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
            seidel_coeffs_x[7] * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
            seidel_coeffs_x[8] * 6 * rho ** 5

    ws_model['x'] = ws_model['x'] + model
    return ws_model


def tertiary_fit_date(date, ws_data, ws_model, fiber):
    path_optical = 'data/aberrations_coefficients/tertiary_coefficients_timeseries/' + date + '/'
    if not os.path.exists(path_optical):
        os.mkdir(path_optical)
    file_out = open(path_optical + 'tertiary_coefs_' + str(fiber) + '.dat', 'w')
    file_out.write('coord,a0,a1,a2,a3,a4,a5,a6,a7,a8\n')

    # Correction in X
    tertiary_coeffs_x = optical_aberrations.correct_dyn_tertiary(ws_data, ws_model, 'x', fiber, date)
    #seidel_coeffs_x = optical_aberrations.correct_seidel(ws_data, ws_model, 'x', fiber, date)
    file_out.write('x,%f,%f,%f,%f,%f\n' % (tertiary_coeffs_x[0],
                                           tertiary_coeffs_x[1],
                                           tertiary_coeffs_x[2],
                                           tertiary_coeffs_x[3],
                                           tertiary_coeffs_x[4]
                                            ))
    file_out.close()

    xprime = (ws_model['x'].values - 4250. / 2) / (4250 / 2)
    yprime = (ws_model['y'].values - 4200. / 2) / (4200 / 2)

    rho = np.sqrt(xprime ** 2 + yprime ** 2)
    theta = np.arctan2(yprime, xprime)

    model = tertiary_coeffs_x[0] * 8 * rho ** 7 + \
            tertiary_coeffs_x[1] * (7 * rho ** 6 * np.cos(theta) + rho ** 7 * np.sin(theta)) + \
            tertiary_coeffs_x[2] * (6 * rho ** 5 * np.cos(theta) ** 2 + 2 * rho ** 6 * np.cos(theta) * np.sin(theta)) + \
            tertiary_coeffs_x[3] * (5 * rho ** 4 * np.cos(theta) ** 3 + 3 * rho ** 5 * np.cos(theta) ** 2 * np.sin(theta)) + \
            tertiary_coeffs_x[4] * (4 * rho ** 3 * np.cos(theta) ** 4 + 4 * rho ** 4 * np.cos(theta) ** 3 * np.sin(theta))

    ws_model['x'] = ws_model['x'] + model
    return ws_model


def poly_fit_date(date, ws_data, ws_model, fiber):
    poly_coeffs_x = poly_fit.correct_dyn(ws_data, ws_model, fiber, date)
    polymodel = poly_fit.function(poly_coeffs_x, ws_model)
    ws_model['x'] = ws_model['x'] + polymodel

    residuals_model = ws_data['posm'].values - ws_model['x'].values
    residuals_data = ws_data['posm'].values - ws_data['posc'].values
    plt.plot(ws_data['posm'].values, residuals_data, 'k.', alpha=0.5)
    plt.plot(ws_data['posm'].values, residuals_model, 'k.', alpha=0.5)
    plt.show()
    rms = np.sqrt(np.sum(residuals_data ** 2) / len(residuals_data))
    rms2 = np.sqrt(np.sum(residuals_model ** 2) / len(residuals_model))
    print(rms, rms2)





def chromatic_model_load(ws_model, fiber):
    if fiber == 'A':
        fib = 'a'
    elif fiber == 'B':
        fib = 'b'
    else:
        print('Wrong fiber!')
    path_chromatic = 'data/aberrations_coefficients/chromatic/'
    coefsx = pd.read_csv(path_chromatic + 'chrome_coeffs_' + str(fib) + '.dat', sep=',')
    a0x = coefsx['a0'].values[0]
    a1x = coefsx['a1'].values[0]
    a2x = coefsx['a2'].values[0]
    a3x = coefsx['a3'].values[0]
    chromatic_coeffs_x = [a0x, a1x, a2x, a3x]
    chromatic_model_x = chromatic_aberrations.function(np.array(ws_model['wave']).astype(np.float), chromatic_coeffs_x)
    ws_model['x'] = ws_model['x'] + chromatic_model_x
    return ws_model


def chromatic_model_load_date(date, ws_model, fiber):
    if fiber == 'A':
        fib = 'a'
    elif fiber == 'B':
        fib = 'b'
    else:
        print('Wrong fiber!')
    path_chromatic = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/'+date+'/'
    coefsx = pd.read_pickle(path_chromatic + 'best_fit_pars_' + str(fib) + '.pkl')
    a0x = coefsx['c0']
    a1x = coefsx['c1']
    a2x = coefsx['c2']
    a3x = coefsx['c3']
    chromatic_coeffs_x = [a0x, a1x, a2x, a3x]
    chromatic_model_x = chromatic_aberrations.function(np.array(ws_model['wave']).astype(np.float), chromatic_coeffs_x)
    ws_model['x'] = ws_model['x'] + chromatic_model_x
    return ws_model

def optical_model_load(ws_model, fiber):
    path_optical = 'data/aberrations_coefficients/optical/'
    if fiber == 'A':
        fib = 'a'
    elif fiber == 'B':
        fib = 'b'
    else:
        fib = 'c'

    coefs = pd.read_csv(path_optical + 'seidel_aberrations_coefs_' + str(fib) + '.dat', sep=',')

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
    print(ws_model)
    ws_model_norm = CCD_vis.pix2mm_model(ws_model, 'x')
    rho, theta = cart2pol(ws_model_norm['x'].astype(np.float), ws_model_norm['y'].astype(np.float))
    seidel_aberrations_x = optical_aberrations.function_seidel(coefs_x, rho, theta)
    ws_model['x'] = ws_model['x'] + seidel_aberrations_x

    return ws_model


def optical_model_load_date(date, ws_model, fiber):
    path_optical = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + date + '/'
    if fiber == 'A':
        fib = 'a'
    elif fiber == 'B':
        fib = 'b'
    else:
        fib = 'c'

    coefs = pd.read_csv(path_optical + 'seidel_aberrations_coefs_' + str(fib) + '.dat', sep=',')

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
    print(ws_model)
    ws_model_norm = CCD_vis.pix2mm_model(ws_model, 'x')
    rho, theta = cart2pol(ws_model_norm['x'].astype(np.float), ws_model_norm['y'].astype(np.float))
    seidel_aberrations_x = optical_aberrations.function_seidel(coefs_x, rho, theta)
    ws_model['x'] = ws_model['x'] + seidel_aberrations_x

    return ws_model


def joint_fit_date(date, data, model, fiber):
    path_joint = 'data/aberrations_coefficients/joint_coefficients_timeseries/' + date + '/'
    if not os.path.exists(path_joint):
        os.mkdir(path_joint)
    joint_coeffs_x = joint_aberrations.correct_dyn(data, model, fiber, date)
    aberrations_model = joint_aberrations.function(joint_coeffs_x, model)
    residuals = data['posm'].values - model['x'].values
    plt.plot(data['posm'].values, residuals, 'k.')
    plt.plot(data['posm'].values, aberrations_model, 'r+')
    plt.show()
    plt.clf()

    model['x'] = model['x'] + aberrations_model
    residuals = data['posm'].values - model['x'].values

    plt.plot(data['posm'].values, residuals, 'k.')
    plt.show()
    plt.clf()


if __name__ == '__main__':

    print('test')
