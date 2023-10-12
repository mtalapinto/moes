import numpy as np
import utils
import json
import pandas as pd
import os
from optics import parameters
from optics import vis_spectrometer
from optics import env_data
import matplotlib.pyplot as plt
import matplotlib
import dynesty
import dyplot
import corner
import pickle
import math
import warnings
# matplotlib.use('Qt4agg')
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def load_coeffs(date, fib):
    path_chromatic = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/' + str(date) + '/'
    file_chromatic_coeffs = pd.read_csv(path_chromatic + 'chrome_coeffs_' + str(fib) + '.dat', sep=',')
    a0 = file_chromatic_coeffs['a0'].values[0]
    a1 = file_chromatic_coeffs['a1'].values[0]
    a2 = file_chromatic_coeffs['a2'].values[0]
    a3 = file_chromatic_coeffs['a3'].values[0]

    return a0, a1, a2, a3


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


def function(x, coefs):
    # return coefs[0] + coefs[1]*x + coefs[2]*x**2 + coefs[3]*x**3
    return coefs[0] * x ** 2 + coefs[1] + coefs[2] * x ** -2 + coefs[3] * x ** -4  # + coefs[4]*x**-6 + coefs[5]*x**-8


def correct_dyn(ws_data, ws_model, coord, fiber, date):
    x = ws_model['wave'].values  # wavelength
    if coord == 'x':
        y = ws_data['posm'].values - ws_model['x'].values
    else:
        y = ws_data['posmy'].values - ws_model['y'].values  # y coordinate


    def prior(cube):
        cube[0] = utils.transform_uniform(cube[0], -10., 10.)
        cube[1] = utils.transform_uniform(cube[1], -10., 10.)
        cube[2] = utils.transform_uniform(cube[2], -10., 10.)
        cube[3] = utils.transform_uniform(cube[3], -10., 10.)
        return cube

    def loglike(cube):
        # Extract parameters:
        a0, a1, a2, a3 = cube[0], cube[1], cube[2], cube[3]
        # Generate model:
        model = a0 * x ** 2 + a1 + a2 * x ** -2 + a3 * x ** -4  # + a4*x**-6 + a5*x**-8
        # Evaluate the log-likelihood:
        ndata = len(y)
        sigma_fit = 0.001
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit ** 2) + \
                        (-0.5 * ((y - model) / sigma_fit) ** 2).sum()

        return loglikelihood

    n_params = 4
    outdir = 'data/aberrations_coefficients/chromatic_coefficients_timeseries/'+date+'/'

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
    a0_end = a0
    a1_end = a1
    a2_end  = a2
    a3_end = a3
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
    outdata['lnZ'] = lnZ
    outdata['lnZ_err'] = lnZerr

    pickle.dump(outdata, open(outdir+'best_fit_pars_'+str(fiber)+'.pkl', 'wb'))
    print('Chromatic correction file written...')
    return a0_end, a1_end, a2_end, a3_end
