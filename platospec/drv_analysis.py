from __future__ import division, print_function
from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
#import pymultinest
import utils
import dynesty
from dynesty import plotting as dyplot
import glob
import sys
# Python 3 compatability
from six.moves import range
import sampling
# system functions that are always useful to have
import time, sys, os
from numpy import linalg
import matplotlib
#matplotlib.use('TKAgg')
from mpl_toolkits.mplot3d import Axes3D
import math
# seed the random number generator
np.random.seed(5647)
from matplotlib import rcParams
import ccf


def do_histo_drv_instrument(instr, sn):
    if instr == 'carmenes':
        datadir = '/data/matala/luthien/'+str(instr)+'/ccf/ns'+str(int(sn))+'/'
    else:
        datadir = '/data/matala/luthien/'+str(instr)+'/data/ccf/ns'+str(int(sn))+'/'
    ndets = len(glob.glob(datadir+'*'))
    outdir = '/data/matala/moes/'+str(instr)+'/results/ns' + str(sn) + '/'
    import plots
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in range(ndets):
        print(i)
        rvs = np.arange(-10000, 10001, 100)
        rvdiff, samp = [], []
        for rv in rvs:
            drvs = np.arange(4.5, 12.5, 0.25)
            drvout = []
            for drv in drvs:
                drvout.append(ccf.ccf_gauss_fit_drv(rv, i, 'simple', drv, sn, instr))
            rvdiff.append(min(drvout))
            samp.append(sampling.sampling_maps_det('platospec', i))

        rvdiffout = pd.DataFrame()
        rvdiffout['drv'] = rvdiff
        rvdiffout['samp'] = samp
        rvdiffout.to_csv(outdir+'drv_ccd_' + str(i) + '_simple' + '.tsv', index=False, sep =',')


def do_histo_drv_instrument_full(instr, sn):
    if instr == 'carmenes':
        datadir = '/data/matala/luthien/' + str(instr) + '/ccf/ns' + str(int(sn)) + '_full/'
    else:
        datadir = '/data/matala/luthien/' + str(instr) + '/data/ccf/ns' + str(int(sn)) + '_full/'
    ndets = len(glob.glob(datadir + '*'))
    outdir = '/data/matala/moes/' + str(instr) + '/results/ns' + str(sn) + '_full/'
    import plots
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in range(ndets):
        print(i)
        rvs = np.arange(-10000, 10001, 100)
        rvdiff, samp = [], []
        for rv in rvs:
            drvs = np.arange(4.5, 12.5, 0.25)
            drvout = []
            for drv in drvs:
                drvout.append(ccf.ccf_gauss_fit_drv(rv, i, 'full', drv, sn, instr))
            rvdiff.append(min(drvout))
            samp.append(sampling.sampling_maps_det_full('platospec', i))

        rvdiffout = pd.DataFrame()
        rvdiffout['drv'] = rvdiff
        rvdiffout['samp'] = samp
        rvdiffout.to_csv(outdir + 'drv_ccd_' + str(i) + '_simple' + '.tsv', index=False, sep=',')


def drv_full_plot(instr):
    basedir = "/data/matala/moes/" + str(instr) + "/"
    datadir = '/data/matala/luthien/' + str(instr) + '/'
    if instr == 'platospec':
        pixdir = datadir + 'data/pix_exp/'
        title = 'PLATOSpec'
        fcam = 240.
        fcol = 876.
        slit = 100
        x_um = 2048 * 13.5
        y_um = 2048 * 13.5
        omin = 68
        omax = 122
        pixarray = np.arange(4.5, 19.6, 1.5)
        # n = len(pixarray)
        n = 10
        imccd = slit * fcam / fcol
        rv = 0

    simplefiles = glob.glob('results/ns0/*')
    fullfiles = glob.glob('results/ns0_full/*')
    sims, simsig = [], []
    fulls, fullsig = [], []
    plt.figure(figsize=[8, 4])
    for f in simplefiles:
        data = pd.read_csv(f, sep=',')  # , names=['drv', 'samp'])
        sims.append(np.mean(data['samp'].values))
        simsig.append(np.std(data['drv'].values))
        # plt.plot(np.mean(data['samp'].values), np.std(data['drv'].values), 'bo', label='no tilt')

    for f in fullfiles:
        data = pd.read_csv(f, sep=',')  # , names=['drv', 'samp'])
        fulls.append(np.mean(data['samp'].values))
        fullsig.append(np.std(data['drv'].values))

        # plt.plot(np.mean(data['samp'].values), np.std(data['drv'].values), 'ro', label='with tilt')
    plt.plot(sims, simsig, 'bo', label='no tilt')
    plt.plot(fulls, fullsig, 'ro', label='with tilt')
    plt.xlabel(r'$s$ (pix)')
    plt.ylabel(r'$\sigma_{RV}$ (m/s)')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(basedir + 'full_ccf_comparison.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':

    #do_histo_drv_instrument('platospec', 1)
    drv_full_plot('platospec')


