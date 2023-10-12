import pandas as pd
import numpy as np
import utils
import json
from optics import parameters
from optics import vis_spectrometer
from optics import env_data
import matplotlib.pyplot as plt
import aberration_corrections
from optics import CCD_vis
import corner
import dynesty
import pickle
import math
import warnings
import os
# matplotlib.use('Qt4agg')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def resample_equal(samples, weights, rstate=None):
    """
    Resample a new set of points from the weighted set of inputs
    such that they all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
    with `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Set of unequally weighted samples.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New set of samples with equal weights.

    Examples
    --------
#    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
#    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
#    >>> utils.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in `Hol, Schon, and
    Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.

    """

    if rstate is None:
        rstate = np.random

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        # Guarantee that the weights will sum to 1.
        warnings.warn("Weights do not sum to 1 and have been renormalized.")
        weights = np.array(weights) / np.sum(weights)

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = np.zeros(nsamples, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


def get_sum(vec):
    fvec = np.sort(vec)
    fval = np.median(fvec)
    nn = int(np.around(len(fvec) * 0.15865))
    vali, valf = fval - fvec[nn], fvec[-nn] - fval
    return fval, vali, valf


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)
    return rho, theta

def function_seidel(ws_model, seidel_coeffs_x):
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

    return model

def function_seidel_old(coeffs, rho, theta):
    seidel_model = coeffs[0] * 4 * rho ** 3 + \
                   coeffs[1] * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
                   coeffs[2] * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
                   coeffs[3] * 2 * rho + \
                   coeffs[4] * (np.cos(theta) - rho * np.sin(theta)) + \
                   coeffs[5] * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta)) + \
                   coeffs[6] * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
                   coeffs[7] * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
                   coeffs[8] * 6 * rho ** 5
    '''
    seidel_model = coeffs[0] * 2 + \
                   coeffs[1] * 2 * rho / np.cos(theta) + \
                   coeffs[2] * 4 * rho * np.cos(theta) + \
                   coeffs[3] * 4 * rho ** 2 + \
                   coeffs[4] * 6 * rho ** 2 * np.cos(theta) ** 2 + \
                   coeffs[5] * 4 * rho ** 3 / np.cos(theta) + \
                   coeffs[6] * 6 * rho ** 3 * np.cos(theta) + \
                   coeffs[7] * 6 * rho ** 4 + \
                   coeffs[8] * 6 * rho ** 5 / np.cos(theta)
    '''
    return seidel_model


def function_seidel_aberrations(coeffs, rho, theta):
    seidel_model = coeffs[0] + \
                   coeffs[1] * 2 * rho * np.cos(theta) + \
                   coeffs[2] * 2 * rho * np.cos(theta) * np.cos(2 * theta) + \
                   coeffs[3] * rho ** 2 * (1 + 2 * np.cos(theta) ** 2) + \
                   coeffs[4] * 3 * rho ** 2 * np.cos(theta) ** 2 + \
                   coeffs[5] * 4 * rho ** 3 * np.cos(theta) + \
                   coeffs[6] * 2 * rho ** 3 * np.cos(theta) * (1 + np.cos(theta) ** 2) + \
                   coeffs[7] * rho ** 4 * (4 * np.cos(theta) ** 2 + 1) + \
                   coeffs[8] * 6 * rho ** 5 * np.cos(theta)

    return seidel_model


def correct_seidel_dyn(date, ws_data, ws_model, coord, fiber):
    if coord == 'x':
        y = ws_data['posm'].values - ws_model['x'].values
        sigma_fit = np.median(ws_data['posme'].values)
    else:
        y = ws_data['posmy'] - ws_model['y']  # y coordinate

    #wsa_model_norm = CCD_vis.pix2mm_model(ws_model, coord)
    ws_data_norm = CCD_vis.pix2mm_data(ws_data, coord)
    rho, theta = cart2pol(ws_data_norm['x_norm'].values, ws_data_norm['y_norm'].values)
    plt.plot(ws_data['posm'].values, y, 'k.')
    print(len(ws_data))
    plt.show()
    plt.clf()
    def prior(cube):
        delta = 1.e1
        cube[0] = utils.transform_uniform(cube[0], -delta, delta)
        cube[1] = utils.transform_uniform(cube[1], -delta, delta)
        cube[2] = utils.transform_uniform(cube[2], -delta, delta)
        cube[3] = utils.transform_uniform(cube[3], -delta, delta)
        cube[4] = utils.transform_uniform(cube[4], -delta, delta)
        cube[5] = utils.transform_uniform(cube[5], -delta, delta)
        cube[6] = utils.transform_uniform(cube[6], -delta, delta)
        cube[7] = utils.transform_uniform(cube[7], -delta, delta)
        cube[8] = utils.transform_uniform(cube[8], -delta, delta)
        return cube

    def loglike(cube):

        # Extract parameters:
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], \
                                             cube[8]

        # Generate model:
        model = a0 * 4 * rho ** 3 + \
                a1 * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
                a2 * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
                a3 * 2 * rho + \
                a4 * (np.cos(theta) - rho * np.sin(theta)) + \
                a5 * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(
            theta)) + \
                a6 * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
                a7 * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
                a8 * 6 * rho ** 5

        # Evaluate the log-likelihood:
        ndata = len(rho)
        #sigma_fit = 0.001
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit ** 2) + (
                -0.5 * ((y - model) / sigma_fit) ** 2).sum()

        return loglikelihood

    n_params = 9
    outdir = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + date + '/'

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    dsampler = dynesty.DynamicNestedSampler(
        loglike,
        prior,
        ndim=n_params)
    dsampler.run_nested(nlive_init=500, nlive_batch=500)
    results = dsampler.results
    samples = results['samples']
    # Get weighted posterior:
    weights = np.exp(results['logwt'] - results['logz'][-1])
    posterior_samples = resample_equal(results.samples, weights)
    # Get lnZ:
    lnZ = results.logz[-1]
    lnZerr = results.logzerr[-1]
    #corner.corner(samples)
    #plt.show()
    a0, a0up, a0lo = get_sum(posterior_samples[:, 0])
    a1, a1up, a1lo = get_sum(posterior_samples[:, 1])
    a2, a2up, a2lo = get_sum(posterior_samples[:, 2])
    a3, a3up, a3lo = get_sum(posterior_samples[:, 3])
    a4, a4up, a4lo = get_sum(posterior_samples[:, 4])
    a5, a5up, a5lo = get_sum(posterior_samples[:, 5])
    a6, a6up, a6lo = get_sum(posterior_samples[:, 6])
    a7, a7up, a7lo = get_sum(posterior_samples[:, 7])
    a8, a8up, a8lo = get_sum(posterior_samples[:, 8])

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

    outdata['lnZ'] = lnZ
    outdata['lnZ_err'] = lnZerr

    pickle.dump(outdata, open(outdir + 'best_fit_pars_'+str(fiber)+'.pkl', 'wb'))
    print('\n')
    return a0, a1, a2, a3, a4, a5, a6, a7, a8


def correct_seidel(ws_data, ws_model, coord, fiber, date):
    if coord == 'x':
        y = ws_data['posm'].values - ws_model['x'].values
    else:
        y = ws_data[:, 5] - ws_model[:, 3]  # y coordinate

    wsa_model_norm = CCD_vis.pix2mm_model(ws_model, coord)
    rho, theta = cart2pol(wsa_model_norm['x_norm'].astype(np.float), wsa_model_norm['y_norm'].astype(np.float))

    def prior(cube, ndim, nparams):

        delta = 1.e3
        cube[0] = utils.transform_uniform(cube[0], -delta, delta)
        cube[1] = utils.transform_uniform(cube[1], -delta, delta)
        cube[2] = utils.transform_uniform(cube[2], -delta, delta)
        cube[3] = utils.transform_uniform(cube[3], -delta, delta)
        cube[4] = utils.transform_uniform(cube[4], -delta, delta)
        cube[5] = utils.transform_uniform(cube[5], -delta, delta)
        cube[6] = utils.transform_uniform(cube[6], -delta, delta)
        cube[7] = utils.transform_uniform(cube[7], -delta, delta)
        cube[8] = utils.transform_uniform(cube[8], -delta, delta)

    def loglike(cube, ndim, nparams):

        # Extract parameters:
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], \
                                             cube[8]

        # Generate model:
        model = a0 * 4 * rho ** 3 + \
                a1 * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
                a2 * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
                a3 * 2 * rho + \
                a4 * (np.cos(theta) - rho * np.sin(theta)) + \
                a5 * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta)) + \
                a6 * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
                a7 * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
                a8 * 6 * rho ** 5

        # Evaluate the log-likelihood:
        ndata = len(rho)
        sigma_fit = 0.01
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit ** 2) + (
                -0.5 * ((model - y) / sigma_fit) ** 2).sum()

        return loglikelihood

    n_params = 9
    out_file = '/luthien/carmenes/vis/optical/' + str(date) + '/ns_seidel_aberrations_' + str(coord) + '_' + str(
        fiber) + '_'

    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points=1000, outputfiles_basename=out_file, resume=False,
                    verbose=False)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    do_outstats(output, date, fiber, coord)
    a0_end = np.mean(mc_samples[:, 0])
    a1_end = np.mean(mc_samples[:, 1])
    a2_end = np.mean(mc_samples[:, 2])
    a3_end = np.mean(mc_samples[:, 3])
    a4_end = np.mean(mc_samples[:, 4])
    a5_end = np.mean(mc_samples[:, 5])
    a6_end = np.mean(mc_samples[:, 6])
    a7_end = np.mean(mc_samples[:, 7])
    a8_end = np.mean(mc_samples[:, 8])
    print('Final optical aberration coefficients for date ' + str(date))
    print(a0_end, a1_end, a2_end, a3_end, a4_end, a5_end, a6_end, a7_end, a8_end)
    print('\n')
    return a0_end, a1_end, a2_end, a3_end, a4_end, a5_end, a6_end, a7_end, a8_end


def do_outstats(output, date, fiber, coord):
    # mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    outstats = output.get_stats()
    print('\n')
    # print(outstats.keys())
    marginals = outstats['marginals']
    modes = outstats['modes']
    maximum = modes[0]['maximum']
    mean = modes[0]['mean']
    labels = ['Spherical', 'Coma', 'Astigmatism', 'Field curvature', 'Distortion', 'Elliptical coma', '2nd Astigmatism',
              '2nd Coma', '2nd Spherical']
    # labels = ['Distortion', 'Field curvature', '1st Astigmatism', '1st Coma', '1st Coma', '1st Spherical', '2nd Astigmatism',
    #          '2nd Astigmatism', '2nd Spherical', '2nd Coma', '2nd Coma', '2nd Spherical']
    ndim = len(marginals)
    sigup, siglo = [], []
    for i in range(ndim):
        sigma = marginals[i]['1sigma']
        sigmaup = sigma[0]
        sigmalo = sigma[1]
        sigup.append(sigmaup)
        siglo.append(sigmalo)
        # print('\n \n')

    logL = modes[0]['local log-evidence']
    logL_sigma = modes[0]['local log-evidence error']
    outdata = pd.DataFrame()
    outdata['logL'] = np.full(ndim, logL)
    outdata['logL_sigma'] = np.full(ndim, logL_sigma)
    outdata['aberrations'] = labels
    outdata['mode'] = maximum
    outdata['mean'] = mean
    outdata['1sigma_up'] = sigup
    outdata['1sigma_lo'] = siglo
    outdir = '/luthien/carmenes/vis/optical/' + str(date) + '/'
    outdata.to_csv(outdir + str(date) + '_optical_' + str(fiber) + '_' + str(coord) + '.tsv', index=False)
    # outL.to_csv(outdir + str(date) + '_optical_logL_' + str(fiber) + '.tsv', index=False)

    # corner.corner(mc_samples, labels=labels)
    # plt.show()
    # print(logL, logL_sigma)
    # print(modes[0])
    # print(maximum)

    # print(modes[0]['mean'])
    # print('\n \n')


def GP_example():
    # Generate random dataset
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(80)
    print(X)
    print(y)
    # Define Gaussian Process kernel (RBF kernel)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    # Initialize Gaussian Process Regressor
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit the Gaussian Process to the data
    gp.fit(X, y)

    # Generate test data
    x_pred = np.atleast_2d(np.linspace(0, 5, 1000)).T

    # Predict mean and standard deviation of the Gaussian Process at test points
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, c='r', marker='x', label='Training Data')
    plt.plot(x_pred, y_pred, 'b', label='Predicted Mean')
    plt.fill_between(x_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue',
                     label='95% Confidence Interval')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()

def correct_dyn(ws_data, ws_model, coord, fiber, date):
    x = ws_model['wave'].values  # wavelength
    if coord == 'x':
        y = ws_data['posm'].values - ws_model['x'].values
    else:
        y = ws_data['posmy'].values - ws_model['y'].values  # y coordinate

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
        return cube

    def loglike(cube):
        # Extract parameters:
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], cube[8]

        # Generate model:
        model = a0 * 4 * rho ** 3 + \
                a1 * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
                a2 * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
                a3 * 2 * rho+ \
                a4 * (np.cos(theta) - rho * np.sin(theta)) + \
                a5 * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta))+ \
                a6 * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
                a7 * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
                a8 * 6 * rho ** 5
        # Evaluate the log-likelihood:
        ndata = len(y)
        sigma_fit = 0.001
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit ** 2) + \
                        (-0.5 * ((y - model) / sigma_fit) ** 2).sum()

        return loglikelihood

    n_params = 9
    outdir = 'data/aberrations_coefficients/optical_coefficients_timeseries/'+date+'/'

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

    outdata['lnZ'] = lnZ
    outdata['lnZ_err'] = lnZerr

    pickle.dump(outdata, open(outdir+'best_fit_pars_'+str(fiber)+'.pkl', 'wb'))
    print('Optical correction file written...')

    # Test plot for the fit
    '''
    plt.clf()
    plt.plot(ws_data['posm'], y, 'k.')
    print(np.sqrt(np.sum(y**2)/len(y)))

    model = a0 * 4 * rho ** 3 + \
            a1 * (3 * rho ** 2 * np.cos(theta) + rho ** 3 * np.sin(theta)) + \
            a2 * (2 * rho * np.cos(theta) + 2 * rho ** 2 * np.cos(theta) * np.sin(theta)) + \
            a3 * 2 * rho + \
            a4 * (np.cos(theta) - rho * np.sin(theta)) + \
            a5 * (3 * rho ** 2 * np.cos(theta) ** 3 + 3 * rho ** 3 * np.cos(theta) ** 2 * np.sin(theta)) + \
            a6 * (4 * rho ** 3 * np.cos(theta) ** 2 + rho ** 4 * 2 * np.sin(theta) * np.cos(theta)) + \
            a7 * (5 * rho ** 4 * np.cos(theta) + rho ** 5 * np.sin(theta)) + \
            a8 * 6 * rho ** 5
    ws_model['x'] = ws_model['x'] + model
    plt.plot(ws_data['posm'], model, 'r.')
    plt.show()
    residuals = ws_data['posm'].values - ws_model['x'].values
    print(np.sqrt(np.sum(residuals ** 2) / len(y)))
    plt.clf()
    plt.plot(ws_data['posm'], residuals, 'k.')
    plt.show()
    '''
    return a0, a1, a2, a3, a4, a5, a6, a7, a8


def correct_dyn_tertiary(ws_data, ws_model, coord, fiber, date):
    #x = ws_model['wave'].values  # wavelength
    x = ws_model['order'].values * ws_model['wave']
    y = ws_data['posm'].values - ws_model['x'].values

    xprime = (ws_model['x'].values - 4250. / 2) / (4250 / 2)
    yprime = (ws_model['y'].values - 4200. / 2) / (4200 / 2)

    rho = np.sqrt(xprime ** 2 + yprime ** 2)
    theta = np.arctan2(yprime, xprime)
    #plt.plot(xprime, yprime, 'ro')
    #plt.show()
    #plt.clf()
    #
    def prior(cube):
        cube[0] = utils.transform_uniform(cube[0], -10., 10.)
        cube[1] = utils.transform_uniform(cube[1], -10., 10.)
        cube[2] = utils.transform_uniform(cube[2], -10., 10.)
        cube[3] = utils.transform_uniform(cube[3], -10., 10.)
        cube[4] = utils.transform_uniform(cube[4], -10., 10.)
        cube[5] = utils.transform_uniform(cube[5], -10., 10.)
        #cube[6] = utils.transform_uniform(cube[6], -10., 10.)
        #cube[7] = utils.transform_uniform(cube[7], -10., 10.)
        #cube[8] = utils.transform_uniform(cube[8], -10., 10.)
        #cube[9] = utils.transform_uniform(cube[9], -10., 10.)
        #cube[10] = utils.transform_uniform(cube[10], -10., 10.)
        #cube[11] = utils.transform_uniform(cube[11], -10., 10.)
        #cube[12] = utils.transform_uniform(cube[12], -10., 10.)
        #cube[13] = utils.transform_uniform(cube[13], -10., 10.)

        return cube

    def loglike(cube):
        # Extract parameters:
        a0, a1, a2, a3, a4, a5 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]

        # Generate model:
        model = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5

        # Evaluate the log-likelihood:
        ndata = len(y)
        sigma_fit = 0.001
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit ** 2) + \
                        (-0.5 * ((y - model) / sigma_fit) ** 2).sum()

        return loglikelihood

    n_params = 6
    outdir = 'data/aberrations_coefficients/tertiary_coefficients_timeseries/'+date+'/'

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
    #a6, a6up, a6lo = get_sum(samples[:, 6])
    #a7, a7up, a7lo = get_sum(samples[:, 7])
    #a8, a8up, a8lo = get_sum(samples[:, 8])
    #a9, a9up, a9lo = get_sum(samples[:, 4])
    #a10, a10up, a10lo = get_sum(samples[:, 5])
    #a11, a11up, a11lo = get_sum(samples[:, 6])
    #a12, a12up, a12lo = get_sum(samples[:, 7])
    #a13, a13up, a13lo = get_sum(samples[:, 8])

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
    #outdata['c6'] = a6
    #outdata['c6_up'] = a6up
    #outdata['c6_lo'] = a6lo
    #outdata['c7'] = a7
    #outdata['c7_up'] = a7up
    #outdata['c7_lo'] = a7lo
    #outdata['c8'] = a8
    #outdata['c8_up'] = a8up
    #outdata['c8_lo'] = a8lo
    #outdata['c9'] = a9
    #outdata['c9_up'] = a9up
    #outdata['c9_lo'] = a9lo
    #outdata['c10'] = a10
    #outdata['c10_up'] = a10up
    #outdata['c10_lo'] = a10lo
    #outdata['c11'] = a11
    #outdata['c11_up'] = a11up
    #outdata['c11_lo'] = a11lo
    #outdata['c12'] = a12
    #outdata['c12_up'] = a12up
    #outdata['c12_lo'] = a12lo
    #outdata['c13'] = a13
    #outdata['c13_up'] = a13up
    #outdata['c13_lo'] = a13lo

    outdata['lnZ'] = lnZ
    outdata['lnZ_err'] = lnZerr

    pickle.dump(outdata, open(outdir+'best_fit_pars_'+str(fiber)+'.pkl', 'wb'))
    print('Tertiary correction file written...')

    # Test plot for the fit

    plt.clf()
    plt.plot(ws_data['posm'], y, 'k.')
    print(np.sqrt(np.sum(y**2)/len(y)))

    model = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5

    ws_model['x'] = ws_model['x'] + model
    plt.plot(ws_data['posm'], model, 'r.')
    plt.show()
    residuals = ws_data['posm'].values - ws_model['x'].values
    print(np.sqrt(np.sum(residuals ** 2) / len(y)))
    plt.clf()
    plt.plot(ws_data['posm'], residuals, 'k.')
    plt.show()

    return a0, a1, a2, a3, a4, a5   #, a6, a7, a8, a9, a10, a11, a12, a13


def load_coeffs(date, fib):
    path_chromatic = 'data/aberrations_coefficients/optical_coefficients_timeseries/' + str(date) + '/'
    file_chromatic_coeffs = pd.read_csv(path_chromatic + 'seidel_coefs_' + str(fib) + '.dat', sep=',')
    a0 = file_chromatic_coeffs['a0'].values[0]
    a1 = file_chromatic_coeffs['a1'].values[0]
    a2 = file_chromatic_coeffs['a2'].values[0]
    a3 = file_chromatic_coeffs['a3'].values[0]
    a4 = file_chromatic_coeffs['a4'].values[0]
    a5 = file_chromatic_coeffs['a5'].values[0]
    a6 = file_chromatic_coeffs['a6'].values[0]
    a7 = file_chromatic_coeffs['a7'].values[0]
    a8 = file_chromatic_coeffs['a8'].values[0]

    return a0, a1, a2, a3, a4, a5, a6, a7, a8


if __name__ == '__main__':

    GP_example()