import warnings
import glob
import chromatic_aberrations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.time import Time
import pickle
import optical_aberrations

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
import ws_load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression


def get_sum(vec):
    fvec = np.sort(vec)
    fval = np.median(fvec)
    nn = int(np.around(len(fvec) * 0.15865))
    vali, valf = fval - fvec[nn], fvec[-nn] - fval
    return fval, vali, valf


def joint_function(coeffs, ws_model):
    xprime = (ws_model['x'].values - 4250. / 2) / (4250 / 2)
    yprime = (ws_model['y'].values - 4200. / 2) / (4200 / 2)

    rho = np.sqrt(xprime ** 2 + yprime ** 2)
    theta = np.arctan2(yprime, xprime)
    x = ws_model['wave'].values
    model = (coeffs[0] * 4 * rho ** 3 + \
             coeffs[1] * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
             coeffs[2] * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
             coeffs[3] * 2 * rho + \
             coeffs[4] * (np.cos(theta) - rho * np.sin(theta)) + \
             coeffs[5] * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta)) + \
             coeffs[6] * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
             coeffs[7] * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
             coeffs[8] * 6 * rho ** 5 + \
             coeffs[9] * x ** 2 + \
             coeffs[10] + \
             coeffs[11] * x ** -2 + \
             coeffs[12] * x ** -4)
    return model


def correct_dyn(ws_data, ws_model, fiber, date):
    x = ws_model['wave'].values  # wavelength
    y = ws_data['posm'].values - ws_model['x'].values

    xprime = (ws_model['x'].values - 4250. / 2) / (4250 / 2)
    yprime = (ws_model['y'].values - 4200. / 2) / (4200 / 2)

    rho = np.sqrt(xprime ** 2 + yprime ** 2)
    theta = np.arctan2(yprime, xprime)
    #plt.plot(xprime, yprime, 'ro')
    #plt.show()
    #plt.clf()

    def prior(cube):
        cube[0] = utils.transform_uniform(cube[0], -10., 10.)
        cube[1] = utils.transform_uniform(cube[1], -10., 10.)
        cube[2] = utils.transform_uniform(cube[2], -10., 10.)
        cube[3] = utils.transform_uniform(cube[3], -10., 10.)
        cube[4] = utils.transform_uniform(cube[4], -10., 10.)
        cube[5] = utils.transform_uniform(cube[5], -10., 10.)
        cube[6] = utils.transform_uniform(cube[6], -10., 10.)
        cube[7] = utils.transform_uniform(cube[7], -10., 10.)
        cube[8] = utils.transform_uniform(cube[8], -10., 10.)
        cube[9] = utils.transform_uniform(cube[9], -10., 10.)
        cube[10] = utils.transform_uniform(cube[10], -10., 10.)
        cube[11] = utils.transform_uniform(cube[11], -10., 10.)
        cube[12] = utils.transform_uniform(cube[12], -10., 10.)

        return cube

    def loglike(cube):
        # Extract parameters:
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], cube[8], cube[9], cube[10], cube[1], cube[12]


        # Generate model:
        model = (a0 * 4 * rho ** 3 + \
                a1 * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
                a2 * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
                a3 * 2 * rho+ \
                a4 * (np.cos(theta) - rho * np.sin(theta)) + \
                a5 * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta))+ \
                a6 * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
                a7 * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
                a8 * 6 * rho ** 5 + \
                a9 * x ** 2 + \
                a10 + \
                a11 * x ** -2 + \
                a12 * x ** -4)

        # Evaluate the log-likelihood:
        ndata = len(y)
        sigma_fit = 0.001
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit ** 2) + \
                        (-0.5 * ((y - model) / sigma_fit) ** 2).sum()

        return loglikelihood

    n_params = 13
    outdir = 'data/aberrations_coefficients/joint_coefficients_timeseries/'+date+'/'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Run MultiNest:
    dsampler = dynesty.DynamicNestedSampler(
        loglike,
        prior,
        ndim=n_params
        )
    dsampler.run_nested(nlive_init=500, nlive_batch=500)
    results = dsampler.results
    samples = results['samples']
    # Get weighted posterior:
    weights = np.exp(results['logwt'] - results['logz'][-1])
    #posterior_samples = resample_equal(results.samples, weights)

    # Get lnZ:
    lnZ = results.logz[-1]
    lnZerr = results.logzerr[-1]

    a0, a0up, a0lo = get_sum(samples[:, 0])
    a1, a1up, a1lo = get_sum(samples[:, 1])
    a2, a2up, a2lo = get_sum(samples[:, 2])
    a3, a3up, a3lo = get_sum(samples[:, 3])
    a4, a4up, a4lo = get_sum(samples[:, 4])
    a5, a5up, a5lo = get_sum(samples[:, 5])
    a6, a6up, a6lo = get_sum(samples[:, 6])
    a7, a7up, a7lo = get_sum(samples[:, 7])
    a8, a8up, a8lo = get_sum(samples[:, 8])
    a9, a9up, a9lo = get_sum(samples[:, 9])
    a10, a10up, a10lo = get_sum(samples[:, 10])
    a11, a11up, a11lo = get_sum(samples[:, 11])
    a12, a12up, a12lo = get_sum(samples[:, 12])

    outdata = {}
    outdata['c0'] = a0
    outdata['c0_up'] = a0up
    outdata['c0_lo'] = a0lo
    outdata['c1'] = a1
    outdata['c1_up'] = a1up
    outdata['c1_lo'] = a1lo
    outdata['c2'] = a2
    outdata['c2_up'] = a2up
    outdata['c2_lo'] = a2lo
    outdata['c3'] = a3
    outdata['c3_up'] = a3up
    outdata['c3_lo'] = a3lo
    outdata['c4'] = a4
    outdata['c4_up'] = a4up
    outdata['c4_lo'] = a4lo
    outdata['c5'] = a5
    outdata['c5_up'] = a5up
    outdata['c5_lo'] = a5lo
    outdata['c6'] = a6
    outdata['c6_up'] = a6up
    outdata['c6_lo'] = a6lo
    outdata['c7'] = a7
    outdata['c7_up'] = a7up
    outdata['c7_lo'] = a7lo
    outdata['c8'] = a8
    outdata['c8_up'] = a8up
    outdata['c8_lo'] = a8lo
    outdata['c9'] = a9
    outdata['c9_up'] = a9up
    outdata['c9_lo'] = a9lo
    outdata['c10'] = a10
    outdata['c10_up'] = a10up
    outdata['c10_lo'] = a10lo
    outdata['c11'] = a11
    outdata['c11_up'] = a11up
    outdata['c11_lo'] = a11lo
    outdata['c12'] = a12
    outdata['c12_up'] = a12up
    outdata['c12_lo'] = a12lo

    outdata['lnZ'] = lnZ
    outdata['lnZ_err'] = lnZerr

    pickle.dump(outdata, open(outdir+'best_fit_pars_'+str(fiber)+'.pkl', 'wb'))
    print('Joint correction file written...')
    return a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12