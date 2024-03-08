from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
from scipy.signal import argrelextrema
import pyfits
from astropy.io import fits
import math as mp
from PyAstronomy import pyasl
from astropy.modeling import models
from astropy import units as u
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils import SpectralRegion
from astropy.modeling.polynomial import Chebyshev1D
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Linear1D, BlackBody
from astropy.modeling import models, fitting
import scipy
from astropy.modeling.models import Voigt1D
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.constants import h, k, c
import os
#import pymultinest
from scipy.special import wofz
from scipy.special import voigt_profile



def ccf_1979():
    rvmin = -40.0
    rvmax = 40.0
    drv = 0.1
    ccf = []
    rv_array = []
    sigma_ccf = []
    tempath = 'stellar_spectrum/star_template_orders/'
    shiftpath = 'stellar_spectrum/star_shifted_orders/'
    while rvmin < rvmax:
        print(rvmin)
        ccsum = 0.
        Ntot = 0.
        flux_obs_aux = 0.
        flux_temp_aux = 0.
        omin = 63
        # omax = 104
        omax = 104
        sigma_ccf_o = []
        while omin <= omax:
            ord = omin
            # Load pixelized data and stellar template
            data_template = pd.read_csv(tempath + str(ord) + '_star_spec.dat', sep=',')
            data_shift = pd.read_csv(shiftpath + str(ord) + '_shift_spec.dat', sep=',')
            hirestemp = pd.read_csv('stellar_spectrum/stellar_spectrum_norm.dat', sep=',')

            spectemp = data_template.copy()
            specobs = data_shift.copy()
            specobs['wave'] = specobs['wave'] * 1e4

            # Setting wavelength range
            hirestemp = hirestemp.loc[hirestemp['wave'] < max(spectemp['wave'] * 1e4)]
            hirestemp = hirestemp.loc[hirestemp['wave'] > min(spectemp['wave'] * 1e4)]

            wmin = min(specobs['wave'])
            wmax = max(specobs['wave'])

            # Check template range
            wmin_temp = min(hirestemp['wave'])
            wmax_temp = max(hirestemp['wave'])
            # print(wmin_temp, wmax_temp)

            if wmin < wmin_temp:
                specobs = specobs.loc[specobs['wave'] > wmin_temp]
            else:
                hirestemp = hirestemp.loc[hirestemp['wave'] > wmin]

            if wmax > wmax_temp:
                specobs = specobs.loc[specobs['wave'] < wmax_temp]
            else:
                hirestemp = hirestemp.loc[hirestemp['wave'] < wmax]

            hirestemp = hirestemp.sort_values(by='wave')
            specobs = specobs.sort_values(by='wave')

            tempdata = hirestemp.copy()
            obsdata = specobs.copy()
            tempdata = tempdata.sort_values(by='wave')
            obsdata = obsdata.sort_values(by='wave')
            tempdata['wave_new'] = tempdata['wave'] * (1 + rvmin / 3.e5)
            wminobs = min(obsdata['wave'].values)
            wmaxobs = max(obsdata['wave'].values)
            wmintemp = min(tempdata['wave_new'])
            wmaxtemp = max(tempdata['wave_new'])

            # Re-set wavelength range after RV shift
            if wminobs < wmintemp:
                obsdata = obsdata.loc[specobs['wave'] > wmintemp]
            else:
                # print(wminobs, wmintemp)
                dl = tempdata['wave'].diff()
                dl = dl.dropna()
                dl = np.average(dl)
                tempdata = tempdata.loc[tempdata['wave_new'] > wminobs - 3 * dl]

            if wmaxobs > wmaxtemp:
                obsdata = obsdata.loc[obsdata['wave'] < wmaxtemp]
            else:
                dl = tempdata['wave'].diff()
                dl = dl.dropna()
                dl = np.average(dl)
                tempdata = tempdata.loc[tempdata['wave_new'] < wmaxobs + 3 * dl]

            tempfunc = interpolate.interp1d(tempdata['wave_new'], tempdata['flux_norm'])

            sigma_ccf_o.append(np.sqrt(np.sum(np.sqrt(tempfunc(obsdata['wave'])))))
            cc = tempfunc(obsdata['wave']) * obsdata['flux']
            N = len(cc)
            flux_temp_aux += np.sum(tempfunc(obsdata['wave']) ** 2)
            flux_obs_aux += np.sum(obsdata['flux'].values ** 2)
            ccsum += np.sum(cc)
            Ntot += N
            omin += 1

        sigma_ccf_o = np.array(sigma_ccf_o)
        sigma_ccf.append(np.sum(sigma_ccf_o ** 2))
        rms_temp = np.sqrt(flux_temp_aux / Ntot)
        rms_obs = np.sqrt(flux_obs_aux / Ntot)
        ccf.append(float(ccsum / (Ntot * rms_temp * rms_obs)))
        rv_array.append(float(rvmin))
        rvmin += drv

    plt.clf()
    ccf_data = pd.DataFrame()
    ccf_data['rv'] = rv_array
    ccf_data['ccf'] = ccf
    ccf_data.to_csv('ccf4.dat', index=False)
    ccf_data = ccf_data.loc[ccf_data['ccf'] > ccf_data['ccf'].values[-1]]
    ccf_data['ccf_norm'] = (ccf_data['ccf'] - min(ccf_data['ccf'].values)) / (
                max(ccf_data['ccf'].values) - min(ccf_data['ccf'].values))

    '''
    ccf_ccf = ccf_data['ccf']
    ccf_rv = ccf_data['rv']
    dccf = ccf_ccf.diff()
    dccf.dropna()
    dv = ccf_rv.diff()
    dv.dropna()
    dccfdrv = dccf/drv

    sigma_ccf = np.array(sigma_ccf)
    sigma = np.power(sigma_ccf, 2)*(1/dccfdrv)
    sigma_rv = 1 / (np.sqrt(np.sum(1 / (sigma) ** 2)))
    print(sigma_rv)
    '''
    mean = np.mean(ccf_data['rv'].values)
    sigma = np.std(ccf_data['rv'].values)
    height = max(ccf_data['ccf'])
    # print(linearr)
    popt, pcov = curve_fit(gaus, ccf_data['rv'], ccf_data['ccf_norm'], p0=[height, mean, sigma])
    perr = np.sqrt(np.diag(pcov))
    print(popt)
    print(perr)
    a = 0.06544
    b = 0.0146
    SN = 100
    sigma_rv = b + a * (1.6 + 0.2 * popt[2]) / SN
    print(sigma_rv)
    plt.plot(ccf_data['rv'], ccf_data['ccf_norm'], 'k.')
    plt.plot(ccf_data['rv'], gaus(ccf_data['rv'].values, *popt), 'r-')
    plt.savefig('ccf_test_5.png')
    # plt.show()
    plt.clf()

    return 0


def ccf2(rv, fcam, instr):
    wmin = 0.36
    wmax = 0.72
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    print('Creating MOES spectra for ' + str(instr))

    rvmin = -15. + rv * 1e-3
    rvmax = 15. + rv * 1e-3
    drv = 0.2
    ccf = []
    rv_array = []
    # indir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/'+str(int(rv))+'/'
    # zerodir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam2)) + 'mm/2D/'+str(int(0))+'/'
    indir = "".join(
        ['/home/eduspec/Documentos/moes/', str(instr), '/data/f', str(int(fcam)), 'mm/', str(int(rv)) + '/'])
    zerodir = "".join(
        ['/home/eduspec/Documentos/moes/', str(instr), '/data/f', str(int(fcam)), 'mm/', str(int(0)), '/'])

    print('Calculating CCF, fcam = ', fcam, ', rv = ', rv)
    # ccfoutdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/ccf/'
    # ccfoutdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/'])
    ccfoutdir = "".join(['/home/eduspec/Documentos/moes/platospec/data/f', str(int(fcam)), 'mm/ccf/'])
    if not os.path.exists(ccfoutdir):
        os.makedirs(ccfoutdir)

    fileout = ccfoutdir + 'ccf_' + str(int(rv)) + '.tsv'
    if not os.path.exists(fileout):
        while rvmin <= rvmax:
            ccsum = 0.
            Ntot = 0.
            flux_obs_aux = 0.
            flux_temp_aux = 0.
            ominaux = omin
            # omax = 104
            omaxaux = omax
            print(fcam, rv, rvmin)
            drv_array = []
            while ominaux <= 80:  # omaxaux:
                # Load pixelized data and stellar template
                tempix = pd.read_csv(zerodir + str(int(omin)) + '_2D.tsv', sep=',')
                tempix = tempix.sort_values('xpix')
                stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
                temp = pd.read_csv(stardir + 'P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec',
                                   delim_whitespace=True,
                                   names=['wave', 'flux', 'flux_norm'])

                # temp = pd.read_csv('stellar_spectrum/stellar_spectrum_norm.dat', sep=',')
                temp = temp.sort_values('wave')
                tempaux = temp.copy()
                obs = pd.read_csv(indir + str(int(omin)) + '_2D.tsv', sep=',')

                # Transform wavelength units of observed spectra
                obs['wave'] = obs['wave'] * 1e4

                # we remove 20 pixels on both sides and we set the spectrum that will be used for the CCF
                tempix = tempix.iloc[20:]
                tempix = tempix.iloc[:-20]
                tempix['wave'] = tempix['wave'] * 1e4
                tempix['wave_nom'] = tempix['wave'] * (1 + rv * 1e-3 / 3.e5)
                wmin_temp = min(tempix['wave_nom'].values)
                wmax_temp = max(tempix['wave_nom'].values)

                obs = obs.loc[obs['wave'] < wmax_temp]
                obs = obs.loc[obs['wave'] > wmin_temp]

                # We select wavelength range of the high resolution template
                temp = temp.loc[temp['wave'] < max(tempix['wave'])]
                temp = temp.loc[temp['wave'] > min(tempix['wave'])]
                temp = temp.sort_values('wave')

                # we shift the template by an amount rvmin
                temp['wave_new'] = temp['wave'] * (1 + rvmin / 3.e5)
                tempaux['wave_new'] = tempaux['wave'] * (1 + rvmin / 3.e5)

                if min(temp['wave_new']) <= min(obs['wave']):
                    temp = temp.loc[temp['wave_new'] > min(obs['wave'])]
                else:
                    obs = obs.loc[obs['wave'] > min(temp['wave_new'])]
                    temp = temp.loc[temp['wave_new'] > min(obs['wave'])]

                if max(temp['wave_new']) <= max(obs['wave']):
                    obs = obs.loc[obs['wave'] < max(temp['wave_new'])]
                    temp = temp.loc[temp['wave_new'] < max(obs['wave'])]
                else:
                    temp = temp.loc[temp['wave_new'] < max(obs['wave'])]

                N = len(obs)  # + len(temp)
                tempaux = tempaux.sort_values(by='wave_new')
                tempfunc = interpolate.interp1d(tempaux['wave_new'], tempaux['flux_norm'])
                cc = tempfunc(obs['wave']) * obs['flux']
                flux_temp_aux += np.sum(tempfunc(obs['wave']) ** 2)
                flux_obs_aux += np.sum(obs['flux'].values ** 2)
                ccsum += np.sum(cc)
                Ntot += N

                ominaux += 1

            rms_temp = np.sqrt(flux_temp_aux / Ntot)
            rms_obs = np.sqrt(flux_obs_aux / Ntot)
            ccf.append(float(ccsum / (Ntot * rms_temp * rms_obs)))
            rv_array.append(float(rvmin))
            rvmin += drv

        ccf_data = pd.DataFrame()
        ccf_data['rv'] = rv_array
        ccf_data['ccf'] = ccf
        ccf_data['ccf_norm'] = (ccf_data['ccf'] - min(ccf_data['ccf'].values)) / (
                max(ccf_data['ccf'].values) - min(ccf_data['ccf'].values))

        ccf_data.to_csv(fileout, index=False)
        plt.plot(ccf_data['rv'], ccf_data['ccf'], 'k-')
        # plt.show()
        plt.clf()
    else:
        print('File already created...')
    return 0


def ccf_new(rv, fcam, instr):
    wmin = 0.38
    wmax = 0.68
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax + 1

    rvmin = -15. + rv * 1e-3
    rvmax = 15. + rv * 1e-3
    drv = 0.2
    ccf = []
    rv_array = []
    basedir = '/home/eduspec/Documentos/moes/'
    zerodir = "".join([basedir, str(instr), '/data/f', str(int(fcam)), 'mm/', str(0), '/'])
    indir = "".join([basedir, str(instr), '/data/f', str(int(fcam)), 'mm/', str(int(rv)) + '/'])
    stardir = '/home/eduspec/Documentos/moes/stellar_template/'

    # zerodir = "".join(
    #    ['/home/eduspec/Documentos/moes/', str(instr), '/data/f', str(int(fcam)), 'mm/', str(int(0)), '/'])

    print('Calculating CCF, fcam = ', fcam, ', rv = ', rv)
    # ccfoutdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/ccf/'
    # ccfoutdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/'])
    ccfoutdir = "".join(['/home/eduspec/Documentos/moes/platospec/data/f', str(int(fcam)), 'mm/ccf/'])

    if not os.path.exists(ccfoutdir):
        os.makedirs(ccfoutdir)
    mask = pd.read_csv(stardir + 'g2mask.tsv', delim_whitespace=True, names=['wini', 'wend', 'weight'])
    mask['delta'] = mask['wend'] - mask['wini']
    temp = pd.read_csv(stardir + 'stellar_template.tsv',
                       sep=',')
    temp = temp.sort_values('wave')

    order = 90
    zerodata = pd.read_csv("".join([zerodir, str(order) + '_2D.tsv']), sep=',')
    zerodata['wave'] = zerodata['wave'] * 1e4
    # print(zerodata)

    wmin0 = min(zerodata['wave'].values)
    wmax0 = max(zerodata['wave'].values)
    plt.plot(zerodata['wave'], zerodata['flux'], 'r-', alpha=0.5)
    plt.plot(temp['wave'], temp['flux'], 'k-', alpha=0.5)
    plt.xlim(wmin0, wmax0)
    # plt.show()
    plt.clf()
    # plt.show()
    ccf = []
    rv_array = []
    ccf_i = []
    fileout = ccfoutdir + 'ccf_' + str(int(rv)) + '.tsv'
    if not os.path.exists(fileout):
        while rvmin <= rvmax:

            ominaux = omin
            omaxaux = omax
            print(fcam, rv, rvmin)
            ccf_out = 0.
            while ominaux <= omaxaux:
                # Load pixelized data
                # stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
                # temp = pd.read_csv(stardir + 'P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec',
                #                   delim_whitespace=True,
                #                   names=['wave', 'flux', 'flux_norm'])

                # temp = pd.read_csv('stellar_spectrum/stellar_spectrum_norm.dat', sep=',')
                obs = pd.read_csv(indir + str(int(omin)) + '_2D.tsv', sep=',')
                obs = obs.loc[obs['y'] >= 0]
                obs = obs.loc[obs['y'] <= 2048]

                # Transform wavelength units of observed spectra
                obs['wave'] = obs['wave'] * 1e4
                # Select binary masks per order
                mask = pd.read_csv(stardir + 'g2mask.tsv', delim_whitespace=True, names=['wini', 'wend', 'weight'])
                maskord = mask.loc[mask['wend'] <= max(obs['wave'])]
                maskord = maskord.loc[maskord['wini'] >= min(obs['wave'])]
                # Shift in rv binary mask
                maskord['wend'] = maskord['wend'] * (1 + rvmin / 3.e5)
                maskord['wini'] = maskord['wini'] * (1 + rvmin / 3.e5)

                # Create observed order function
                obswavefunc = interpolate.interp1d(obs['wave'], obs['y'])
                obspixfunc = interpolate.interp1d(obs['y'], obs['wave'])

                # Get pixel position of the binary mas
                maskord['pixini'] = obswavefunc(maskord['wini'])
                maskord['pixend'] = obswavefunc(maskord['wend'])

                # plt.plot(obs['wave'], obs['flux'], 'k-', alpha=0.5)
                # plt.plot(maskord['wini'], np.full(len(maskord), 1), 'ro')
                # plt.plot(maskord['wend'], np.full(len(maskord), 1), 'ro')
                # plt.xlim(6749.75, 6750.50)
                # plt.show()
                # plt.clf()
                ccf_ord = 0.
                wsum_ord = 0.

                for i in range(len(maskord)):
                    # print(mask)
                    mask_weight = maskord['weight'].values[i]
                    pixmin = maskord['pixini'].values[i]
                    pixmax = maskord['pixend'].values[i]

                    obsdata = obs.loc[obs['y'] <= pixmax]
                    obsdata = obsdata.loc[obsdata['y'] >= pixmin]

                    # plt.plot(obs['wave'], obs['flux'], 'k-', alpha=0.5)
                    # plt.plot(maskord['wini'].values[i], 1, 'ro')
                    # plt.plot(maskord['wend'].values[i], 1, 'ro')

                    # plt.xlim(maskord['wini'].values[i] - 0.2, maskord['wend'].values[i] + 0.2)
                    # plt.show()
                    # plt.clf()
                    # print(obsdata)

                    # print(len(obsdata))
                    for k in range(len(obsdata)):
                        minpix_obs = obsdata['y'].values[k] - 0.5
                        maxpix_obs = obsdata['y'].values[k] + 0.5

                        if minpix_obs >= maskord['pixini'].values[i] and maxpix_obs <= maskord['pixend'].values[i]:
                            dpix = 1.
                            # print(dpix, 0)
                            ccfpix = obsdata['flux'].values[k] * mask_weight * dpix
                            ccf_ord += ccfpix
                            wsum_ord += mask_weight ** 2
                        elif minpix_obs >= maskord['pixini'].values[i] and maxpix_obs >= maskord['pixend'].values[i]:
                            dpix = np.abs(minpix_obs - maskord['pixend'].values[i])
                            # print(dpix, 1)
                            ccfpix = obsdata['flux'].values[k] * mask_weight * dpix
                            ccf_ord += ccfpix
                            wsum_ord += mask_weight ** 2
                        elif minpix_obs <= maskord['pixini'].values[i] and maxpix_obs <= maskord['pixend'].values[i]:
                            dpix = np.abs(maskord['pixini'].values[i] - maxpix_obs)
                            # print(dpix, 2)
                            ccfpix = obsdata['flux'].values[k] * mask_weight * dpix
                            ccf_ord += ccfpix
                            wsum_ord += mask_weight ** 2

                        # Check for pixel centers outside the mask
                        if len(obsdata) == 1:
                            if minpix_obs > maskord['pixini'].values[i]:
                                # print(obsdata['y'].values['k'] - 1)
                                outpixdata = obs.loc[obs['y'] == (obsdata['y'].values[k] - 1)]
                                outdpix = np.abs(((outpixdata['y'] + 0.5) - maskord['pixini'].values[i]).values[0])

                                # print(outdpix, mask_weight, outpixdata['flux'].values[0])
                                ccfoutpix = outdpix * mask_weight * outpixdata['flux'].values[0]
                                ccf_ord += ccfoutpix
                                wsum_ord += mask_weight ** 2
                                # plt.plot(outpixdata['y'], 1, 'go')
                                # plt.plot(outpixdata['y'] + 0.5, 1, 'bo')
                                # plt.plot(outpixdata['y'] - 0.5, 1, 'bo')
                            if maxpix_obs < maskord['pixend'].values[i]:
                                outpixdata = obs.loc[obs['y'] == (obsdata['y'].values[k] + 1)]
                                outdpix = np.abs(((outpixdata['y'] - 0.5) - maskord['pixend'].values[i]).values[0])
                                # print(outdpix, mask_weight, outpixdata['flux'].values[0])
                                ccfoutpix = outdpix * mask_weight * outpixdata['flux'].values[0]
                                ccf_ord += ccfoutpix
                                wsum_ord += mask_weight ** 2
                                # plt.plot(outpixdata['y'], 1, 'go')
                                # plt.plot(outpixdata['y'] + 0.5, 1, 'bo')
                                # plt.plot(outpixdata['y'] - 0.5, 1, 'bo')

                        # plt.plot(obs['y'], obs['flux'], 'k-', alpha=0.5)
                        # plt.plot(maskord['pixini'].values[i], 1, 'ro')
                        # plt.plot(maskord['pixend'].values[i], 1, 'ro')
                        # plt.plot(minpix_obs, 1, 'bo')
                        # plt.plot(maxpix_obs, 1, 'bo')
                        # plt.plot(obsdata['y'].values[k], 1, 'ko')
                        # plt.xlim(minpix_obs - 2, maxpix_obs + 2)
                        # plt.show()
                        # plt.clf()
                ccf_out += ccf_ord  # /np.sqrt(wsum_ord)
                ominaux += 1
            # print(ccf_ord)
            ccf_i.append(ccf_out)
            rv_array.append(float(rvmin))
            # rms_temp = np.sqrt(flux_temp_aux / Ntot)
            # rms_obs = np.sqrt(flux_obs_aux / Ntot)
            # ccf.append(float(ccsum / (Ntot * rms_temp * rms_obs)))
            rvmin += drv

        ccf_data = pd.DataFrame()
        ccf_data['rv'] = rv_array
        ccf_data['ccf'] = ccf_i
        ccf_data['ccf_norm'] = (ccf_data['ccf'] - min(ccf_data['ccf'].values)) / (
                max(ccf_data['ccf'].values) - min(ccf_data['ccf'].values))

        ccf_data.to_csv(fileout, index=False)
        plt.clf()
        plt.plot(ccf_data['rv'], ccf_data['ccf_norm'], 'k-')
        plt.show()
        # plt.clf()
    else:
        print('File already created...')

    return 0


def lorentzian(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x ** 2 + gamma ** 2)


def V(x, amp, mean, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return amp * np.real(wofz(((x - mean) + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


def ccf_gaussian_fit(rv, fcam):
    ccfdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/'])
    ccf = pd.read_csv(ccfdir + 'ccf_' + str(int(rv)) + '.tsv', sep=',')

    mean = np.mean(ccf['rv'].values)
    sigma = np.std(ccf['rv'].values)
    height = max(ccf['ccf'])
    offset = 0.1
    # print(linearr)
    popt, pcov = curve_fit(gaus, ccf['rv'], ccf['ccf_norm'], p0=[height, mean, sigma, offset])
    perr = np.sqrt(np.diag(pcov))
    print(popt[1] * 1e3)
    plt.plot(ccf['rv'], ccf['ccf_norm'], 'k-', label='CCF')
    plt.plot(ccf['rv'], gaus(ccf['rv'], *popt), 'r-',
             label=r'Gaussian fit, RV$_{mean}$ = ' + str(np.round(popt[1] * 1.e3, 2)) + '+/- ' + str(
                 np.round(perr[1] * 1e3, 2)) + 'm/s')
    plt.legend()
    plt.ylabel('CCF')
    plt.xlabel('RV (km/s)')
    plt.tight_layout()
    plt.show()
    plt.clf()
    rvmaxfit = np.round(popt[1] * 1.e3, 2)
    diff = rv - rvmaxfit
    return diff

    # print(popt)
    # print(pcov)


def ccf_gauss_ns(rv, fcam):
    ccfdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/'])
    ccf = pd.read_csv(ccfdir + 'ccf_' + str(int(rv)) + '_63_test.tsv', sep=',')
    ccf = ccf.loc[ccf['ccf_norm'] > ccf['ccf_norm'].values[-1]]

    '''
    amp = 11.05
    mean = 0.49
    alpha = 4.3
    gamma = 0.87
    # print(linearr)

    popt, pcov = curve_fit(V, ccf['rv'], ccf['ccf_norm'], p0=[amp, mean, alpha, gamma])
    perr = np.sqrt(np.diag(pcov))

    plt.figure(figsize=(10,4))
    plt.plot(ccf['rv'], ccf['ccf_norm'], 'k-', label='CCF')
    plt.plot(ccf['rv'], V(ccf['rv'], *popt), 'r-',
             label=r'Voigt fit, RV$_{mean}$ = ' + str(np.round(popt[1] * 1.e3, 2)) + ' +/- ' + str(
                 np.round(perr[1] * 1e3, 2)) + ' m/s')
    plt.legend(loc=1)
    plt.ylabel('CCF')
    plt.xlabel('RV (km/s)')
    plt.tight_layout()
    plt.xlim(min(ccf['rv']), max(ccf['rv']))
    ccfplots = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/plots/'])
    plt.savefig(ccfplots+'ccf_'+str(rv)+'.png')
    plt.show()
    plt.clf()
    '''
    x = ccf['rv']
    y = ccf['ccf_norm']

    def prior(cube, ndim, nparams):
        # Prior on RAMSES parameters, sorted by importance
        cube[0] = utils.transform_uniform(cube[0], 0.9, 1.1)  # gaussian height
        cube[1] = utils.transform_uniform(cube[1], (rv * 1e-3) - 1, (rv * 1e-3) + 1)  # mean
        cube[2] = utils.transform_uniform(cube[2], 1., 12.)  # sigma
        cube[3] = utils.transform_uniform(cube[3], -1., 1.)  # offset

    # Define the likelihood:

    def loglike(cube, ndim, nparams):
        # Load parameters
        height = cube[0]  # Voigt mean
        mean = cube[1]  # voigt lorentz hwhm
        sigma = cube[2]  # Voigt gauss hwhm
        offset = cube[3]  # Voigt amplitude
        # Generate model:
        model = gaus(x, height, mean, sigma, offset)
        # Evaluate the log-likelihood:
        sigma_fit_x = np.full(len(y), 1.)

        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + (
                -0.5 * ((model - y) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 4
    outpath = "".join(
        ['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/ns_files/', str(int(rv)), '/'])

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    out_file = outpath + 'ccf_out.dat'

    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points=2000, outputfiles_basename=out_file, resume=False,
                    verbose=False)

    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    # print('Multicarlos optimization duration : %.3f hr' % (float(t2)))
    plt.close()

    amp_mean = np.mean(mc_samples[:, 0])
    mean_mean = np.mean(mc_samples[:, 1])
    sigma_mean = np.mean(mc_samples[:, 2])
    offset_mean = np.mean(mc_samples[:, 3])

    amp = bestfit_params['parameters'][0]
    mean = bestfit_params['parameters'][1]
    sigma = bestfit_params['parameters'][2]
    offset = bestfit_params['parameters'][3]
    print(amp, mean, sigma)
    plt.plot(ccf['rv'], ccf['ccf_norm'], 'k-', label='CCF')
    plt.plot(ccf['rv'], gaus(ccf['rv'], amp, mean, sigma, offset), 'r-', label='Gaussian fit')
    # plt.plot(ccf['rv'], gaus(ccf['rv'], amp_mean, mean_mean, sigma_mean, offset_mean), 'b-', label='Gaussian fit mean')
    plt.show()

    rvdiff = rv - mean * 1.e3
    print(rvdiff)
    return rvdiff


def read_template():
    tempdir = '/home/eduspec/Documentos/moes/stellar_template/'
    basedir = '/home/eduspec/Documentos/moes/platospec/data/f240mm/-5000/'
    moesdata = pd.read_csv(basedir + '75_2D.tsv', sep=',')
    tempdata = pd.read_csv(tempdir + 'stellar_template.tsv', sep=',')
    # model = pyfits.getdata(tempdir+'5750_45_p00p00.ms.fits')
    # hd = pyfits.getheader(tempdir+'5750_45_p00p00.ms.fits')
    # wav = np.arange(model.shape[1]) * hd['CD1_1'] + hd['CRVAL1']
    # flx = model[0]
    # tempout = pd.DataFrame()
    # tempout['wave'] = wav
    # tempout['flx'] = flx
    # tempout.to_csv(tempdir+'stellar_template.tsv', index=False)

    mask = pd.read_csv(tempdir + 'g2mask.tsv', delim_whitespace=True, names=['w0', 'w1', 'weight'])
    w0x, w1x, y = [], [], []
    for i in range(len(mask)):
        w0x.append(mask['w0'].values[i])
        w1x.append(mask['w1'].values[i])
        y.append(1)
        linecen = tempdata.loc[tempdata['wave'] >= mask['w0'].values[i]]
        linecen = linecen.loc[linecen['wave'] <= mask['w1'].values[i]]
        linecen['min'] = linecen.iloc[argrelextrema(linecen.flux.values, np.less_equal, order=3)[0]]['flux']
        linecen = linecen.dropna()
        wcen = linecen['wave'].values[0]
        dl = 0.2
        linedata = tempdata.loc[tempdata['wave'] >= wcen - dl / 2]
        linedata = linedata.loc[linedata['wave'] <= wcen + dl / 2]
        plt.plot(linedata['wave'], linedata['flux'], 'k-')
        plt.plot(mask['w0'].values[i], 0., 'ro')
        plt.plot(mask['w1'].values[i], 0., 'ro')
        plt.show()
        plt.clf()
        print(i, linecen)
        plt.clf()

    plt.xlim(3800, 6800)
    plt.plot(tempdata['wave'], tempdata['flux'], 'k-', alpha=0.5)
    moesdata = moesdata.sort_values('wave')
    plt.plot(moesdata['wave'] * 1e4, moesdata['flux'], 'b-')
    plt.plot(w0x, y, 'ro')
    plt.plot(w1x, y, 'ro')
    plt.show()


def get_good_lines():
    stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    temp = pd.read_csv(stardir + 'P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec',
                       delim_whitespace=True,
                       names=['wave', 'flux', 'flux_norm'])

    temp = temp.loc[temp['wave'] <= 6800]
    temp = temp.loc[temp['wave'] >= 3800]
    temp = temp.sort_values('wave')
    temp['min'] = temp.iloc[argrelextrema(temp.flux_norm.values, np.less_equal, order=3)[0]]['flux_norm']
    print(temp['min'])
    tempmin = temp.dropna()
    wavescen = tempmin['wave']
    plt.plot(temp['wave'], temp['flux_norm'], 'k-')
    plt.plot(tempmin['wave'], tempmin['flux_norm'], 'ro')
    # plt.show()
    plt.clf()

    w, wcenout, R, flux_cont, Im = [], [], [], [], []

    for k in range(len(tempmin)):

        wcen = tempmin['wave'].values[k]
        fluxmin = tempmin['flux_norm'].values[k]
        dlambda = 0.07
        linedata = temp.loc[temp['wave'] < wcen + dlambda]
        linedata = linedata.loc[linedata['wave'] > wcen - dlambda]

        linewings_r = linedata.iloc[-3:]
        linewings_l = linedata.iloc[:3]
        slope_wing_r = (linewings_r['flux_norm'].values[-1] - linewings_r['flux_norm'].values[0]) / (
                    linewings_r['wave'].values[-1] - linewings_r['wave'].values[0])
        slope_wing_l = (linewings_l['flux_norm'].values[-1] - linewings_l['flux_norm'].values[0]) / (
                    linewings_l['wave'].values[-1] - linewings_l['wave'].values[0])

        flux_wing_r = np.mean(linewings_r['flux_norm'])
        std_flux_wing_r = np.std(linewings_r['flux_norm'])
        flux_wing_l = np.mean(linewings_l['flux_norm'])
        std_flux_wing_l = np.std(linewings_l['flux_norm'])
        std_threshold = 0.1
        delta_flux = flux_wing_l - flux_wing_r
        amp_flux = max(linedata['flux_norm']) - min(linedata['flux_norm'])

        if std_flux_wing_r < std_threshold and std_flux_wing_l < std_threshold and np.abs(delta_flux) < 0.1 and len(
                linedata) > 6 and amp_flux > 0.1:
            linedata['min'] = linedata.iloc[argrelextrema(linedata.flux_norm.values, np.less_equal, order=15)[0]][
                'flux_norm']
            linemins = linedata.dropna()
            if 0 < len(linemins) < 2 and min(linemins['wave']) != min(linedata['wave']) and max(
                    linemins['wave']) != max(linedata['wave']):
                # print(flux_wing_r, flux_wing_l, len(linewings_l), slope_wing_l, slope_wing_r, std_flux_wing_r,
                #      std_flux_wing_l)
                x = linedata['wave']
                y = linedata['flux_norm']

                # g_init = models.Gaussian1D(amplitude=-1., mean=wcen, stddev=dlambda)
                # fit_g = fitting.LevMarLSQFitter()
                # g = fit_g(g_init, x, y)
                height = -1
                mean = wcen
                sigma = dlambda
                offset = 1.
                try:
                    popt, pcov = curve_fit(gaus, x, y, p0=[height, mean, sigma, offset])
                    perr = np.sqrt(np.diag(pcov))
                    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                    amp = popt[0]
                    wmean = popt[1]
                    wdiff = wmean - wcen
                    offset = popt[3]
                    print(amp, wmean, wdiff, fwhm)
                    if amp < 0 and np.abs(wdiff) < 0.01 and 0.02 < fwhm < 0.6:
                        w.append(fwhm)
                        wcenout.append(wmean)
                        R.append(wmean / fwhm)
                        flux_cont.append(min(y))
                        Im.append(popt[0] + popt[3])

                        # plt.clf()
                        # gauss_array = np.arange(min(linedata['wave']), max(linedata['wave']), 0.001)
                        # plt.plot(linedata['wave'], linedata['flux_norm'], 'k-')
                        # plt.plot(linemins['wave'], linemins['flux_norm'], 'bo', markersize=5)
                        # plt.plot(linewings_r['wave'], linewings_r['flux_norm'], 'r-')
                        # plt.plot(linewings_l['wave'], linewings_l['flux_norm'], 'r-')
                        # plt.plot(gauss_array, gaus(gauss_array, *popt), 'g-')
                        # plt.plot()
                        # plt.plot(wcen, fluxmin, 'ro')
                        # plt.show()
                        # plt.clf()

                except RuntimeError:
                    print('Error in curve fitting...')

    linelist = pd.DataFrame()
    linelist['w'] = w
    linelist['wave'] = wcenout
    linelist['R'] = R
    linelist['flux_cont'] = flux_cont
    linelist['Im'] = Im
    linelist.to_csv('line_list.tsv', index=False)
    print(len(linelist))
    plt.clf()
    plt.hist(w, bins=30)
    # plt.show()
    plt.clf()
    plt.hist(R, bins=30)
    # plt.show()


def ccf_gauss_fit_v2(rv, fcam):
    # ccfdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/'])
    ccfdir = "".join(['/home/eduspec/Documentos/moes/platospec/data/f', str(int(fcam)), 'mm/ccf/'])
    ccf = pd.read_csv(ccfdir + 'ccf_' + str(int(rv)) + '.tsv', sep=',')
    outfile = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/plots/ccf_gauss_' + str(
        int(rv)) + '.png'
    # ccf = ccf.loc[ccf['ccf_norm'] > ccf['ccf_norm'].values[-1]]
    x = ccf['rv']
    y = ccf['ccf']

    mean = rv * 1e-3
    sigma = np.std(ccf['rv'].values)
    height = min(ccf['ccf'])
    offset = 0.1
    print(mean)
    popt, pcov = curve_fit(gaus, ccf['rv'], ccf['ccf'], p0=[height, mean, sigma, offset])

    # g_init = models.Gaussian1D(amplitude=1., mean=rv*1e-3, stddev=3.)
    # fit_g = fitting.LevMarLSQFitter()
    # g = fit_g(g_init, x, y)
    # print()
    # print(g)
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'ko')
    plt.plot(x, gaus(x, *popt), label='Gaussian fit RV mean = ' + str(popt[1] * 1e3) + ' m/s')
    print(popt[1])
    plt.title('CCF for a nominal RV = ' + str(rv) + ' m/s')
    plt.xlabel('RV (km/s)   ')
    plt.ylabel('CCF')
    plt.legend(loc='best')
    # plt.savefig(outfile)
    plt.show()

    plt.close()
    rvdiff = rv - popt[1] * 1e3
    print(rvdiff)
    return rvdiff


def ccf_voigt(rv, fcam):
    ccfdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/ccf/'
    ccfout = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/ccf/plots/'
    if not os.path.isdir(ccfout):
        os.mkdir(ccfout)
    ccf = pd.read_csv(ccfdir+'ccf_'+str(int(rv))+'.tsv', sep=',')
    amp = 11.05
    mean = float(rv) * 1.e-3
    alpha = 4.3
    gamma = 0.87
    # print(linearr)
    popt, pcov = curve_fit(V, ccf['rv'], ccf['ccf_norm'], p0=[amp, mean, alpha, gamma])
    perr = np.sqrt(np.diag(pcov))

    plt.figure(figsize=(10,4))
    plt.plot(ccf['rv'], ccf['ccf_norm'], 'k-', label='CCF')
    plt.plot(ccf['rv'], V(ccf['rv'], *popt), 'r-',
             label=r'Voigt fit, RV$_{mean}$ = ' + str(np.round(popt[1] * 1.e3, 2)) + ' +/- ' + str(
                 np.round(perr[1] * 1e3, 2)) + ' m/s')
    plt.legend(loc=1)
    plt.ylabel('CCF')
    plt.xlabel('RV (km/s)')
    plt.tight_layout()
    #plt.xlim(min(ccf['rv']), max(ccf['rv']))
    plt.savefig(ccfout+'ccf_voigt_least_squares_'+str(rv)+'.png')
    plt.show()
    plt.clf()
    plt.close()
    deltarv = (rv - popt[1]*1.e3)
    print(deltarv)
    return deltarv


def rv_diff(fcam):
    rvarr = np.arange(-10000, 10001, 250)
    dif_array, dif_array2 = [], []
    for rv in rvarr:
        dif = ccf_gauss_fit_v2(rv, 300)
        dif2 = ccf_gauss_fit_v2(rv, 230)
        dif_array.append(dif)
        dif_array2.append(dif2)
    #print(len(rvarr))

    plt.clf()
    plt.figure(figsize=[8,4])
    n, bins, patches = plt.hist(dif_array, bins=15, color='blue', alpha=0.5)
    rv_array = np.arange(min(bins), max(bins), 0.5)
    #print(rv_array)
    binsize = bins[1] - bins[0]
    bins = bins + binsize/2
    bins = bins[:-1]

    mean = np.mean(bins)
    sigma = np.std(bins)
    amp_hist = np.max(n)+5
    offset = 0.1
    # print(linearr)
    popt, pcov = curve_fit(gaus, bins, n, p0=[amp_hist, mean, sigma, offset])
    #print(popt[2]*2*np.sqrt(2*np.log(2)))
    #print(popt[2])
    perr = np.sqrt(np.diag(pcov))

    g_init = models.Gaussian1D(amplitude=amp_hist, mean=mean, stddev=sigma)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, bins, n)
    #print(g)

    n2, bins2, patches2 = plt.hist(dif_array2, bins=15, color='red', alpha=0.5)
    rv_array2 = np.arange(min(bins2), max(bins2), 0.5)
    # print(rv_array)
    binsize2 = bins2[1] - bins2[0]
    bins2 = bins2 + binsize2 / 2
    bins2 = bins2[:-1]
    mean2 = np.mean(bins2)
    sigma2 = np.std(bins2)
    amp_hist2 = np.max(n2) + 5
    offset2 = 0.1

    popt2, pcov2 = curve_fit(gaus, bins2, n2, p0=[amp_hist2, mean2, sigma2, offset2])

    plt.plot(rv_array2, gaus(rv_array2, *popt2), 'r-', label=r'samp = 2.0 pix')
    plt.plot(rv_array, gaus(rv_array, *popt), 'b-', label=r'samp = 2.6 pix')
    #plt.plot(rv_array, g(rv_array), label='Gaussian2')
    #plt.title(r'Avg. sampling = 2.6, Gaussian fit $\sigma$ = '+str(np.round(popt[2], 2)) + 'm/s')
    plt.ylabel('Number of observations')
    plt.xlabel(r'RV$_{nominal}$ - RV$_{CCF}$ (m/s)')
    plt.legend(loc=2)
    plt.savefig('plots/rv_diff_histo.png')
    #plt.show()

    #print(len(bins), bins, n)
    #print(binsize)
    std = np.std(dif_array)
    std2 = np.std(dif_array2)
    print(std, std2)


def ccf_ns_gauss():
    ccf = pd.read_csv('ccf4.dat', sep=',')
    ccf = ccf.loc[ccf['rv'] < 39.]
    ccf = ccf.loc[ccf['rv'] > -39.]
    ccf = ccf.loc[ccf['ccf'] > ccf['ccf'].values[-1]]
    ccf['ccf_norm'] = (ccf['ccf'] - min(ccf['ccf'].values)) / (max(ccf['ccf'].values) - min(ccf['ccf'].values))

    ccf_func = interpolate.interp1d(ccf['rv'], ccf['ccf_norm'])
    rvarr = np.arange(-38., 38., 0.001)

    ccfarr = ccf_func(rvarr)
    print(ccfarr)
    print(rvarr[np.argmax(ccfarr)])
    mean = np.mean(ccf['rv'].values)
    sigma = np.std(ccf['rv'].values)
    height = max(ccf['ccf'])
    offset = 0.
    # print(linearr)
    popt, pcov = curve_fit(gaus, ccf['rv'], ccf['ccf_norm'], p0=[height, mean, sigma, offset])
    perr = np.sqrt(np.diag(pcov))
    print(perr)
    mean = 0.5  # Voigt mean
    alpha = 5.  # voigt lorentz hwhm
    gamma = 5.  # Voigt gauss hwhm
    amp = 1e2 # Voigt amplitude
    # Generate model:
    # model = gaus(x, height, mean, sigma)
    # model = voigt(x, mean, alpha, gamma)
    # vgt =
    #model =

    plt.plot(ccf['rv'], ccf['ccf_norm'], 'k-', label='CCF')
    plt.plot(ccf['rv'], gaus(ccf['rv'], *popt), 'r-',
             label=r'Gaussian fit, RV$_{mean}$ = ' + str(np.round(popt[1] * 1.e3, 2)) + '+/- ' + str(
                 np.round(perr[1] * 1e3, 2)) + 'm/s')
    plt.legend()
    plt.ylabel('CCF')
    plt.xlabel('RV (km/s)')
    plt.tight_layout()
    plt.show()
    plt.clf()
    x = ccf['rv']
    y = ccf['ccf_norm']

    def prior(cube, ndim, nparams):
        # Prior on RAMSES parameters, sorted by importance
        cube[0] = utils.transform_uniform(cube[0], 0.8, 1.2)  # gaussian height
        cube[1] = utils.transform_uniform(cube[1], -5., 5.)  # gaussian mean
        cube[2] = utils.transform_uniform(cube[2], -10., 10.)  # gaussian sigma
        cube[3] = utils.transform_uniform(cube[3], -5., 5.)  # gaussian offset

    # Define the likelihood:

    def loglike(cube, ndim, nparams):
        # Load parameters
        height = cube[0]  # gauss height
        mean = cube[1]  # gauss mean
        sigma = cube[2]  # gauss sigma
        offset = cube[3]        # Generate model:
        model = gaus(x, height, mean, sigma, offset)

        # Evaluate the log-likelihood:
        sigma_fit_x = np.full(len(y), 1.)

        ndata = len(y)
        loglikelihood = -0.5 * ndata * np.log(2. * np.pi * sigma_fit_x ** 2).sum() + (
                -0.5 * ((model - y) / sigma_fit_x) ** 2).sum()

        return loglikelihood

    n_params = 4
    path = '/home/eduspec/Documentos/moes/fideos_moes/ccf_ns/'

    if not os.path.exists(path):
        os.makedirs(path)
    out_file = path+'ccfout_gauss.dat'

    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points=1000, outputfiles_basename=out_file, resume=True,
                    verbose=False)

    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    bestfit_params = output.get_best_fit()
    # print(type(bestfit_params['parameters']), len(bestfit_params['parameters']))
    # print(bestfit_params['parameters'][0])
    mc_samples = output.get_equal_weighted_posterior()[:, :-1]
    # print('Multicarlos optimization duration : %.3f hr' % (float(t2)))
    plt.close()
    plt.clf()

    height = bestfit_params['parameters'][0]
    mean = bestfit_params['parameters'][1]
    sigma = bestfit_params['parameters'][2]
    offset = bestfit_params['parameters'][3]

    print(mean, gamma*2/2.355)
    plt.clf()
    plt.plot(ccf['rv'], ccf['ccf_norm'], 'k-', label='CCF')
    plt.plot(ccf['rv'], gaus(ccf['rv'], height, mean, sigma, offset), 'r-', label='Gauss')
    plt.show()
    #plt.plot(ccf['rv'], V(ccf['rv'], mean, alpha, gamma, amp), 'r-',
    #         label=r'Gaussian fit - NS, RV$_{mean}$ = ' + str(np.round(popt[1] * 1.e3, 2)) + '+/- ' + str(
    #             np.round(perr[1] * 1e3, 2)) + 'm/s')

    mean = np.mean(mc_samples[:,0])
    sigma = np.std(mc_samples[:, 0])
    print(mean, sigma)
    import corner
    figure = corner.corner(mc_samples)
    plt.show()
    plt.savefig('ccf_gauss.png')


def plot_slits():
    data = pd.read_csv('stellar_spectrum/moes_stellar_spectrum.csv', sep=',')
    order = 75
    orderdata = data.loc[data['order'] == order]
    waves = np.unique(orderdata['wave'].values)
    n = 45
    wave0 = waves[n]
    wave1 = waves[n+5]
    wave2 = waves[n+10]
    wave3 = waves[n+15]
    wave0data = data.loc[data['wave'] == wave0]
    wave1data = data.loc[data['wave'] == wave1]
    wave2data = data.loc[data['wave'] == wave2]
    wave3data = data.loc[data['wave'] == wave3]

    plt.plot(wave0data['y'], wave0data['x'], 'r.', label=wave0)
    plt.plot(wave1data['y'], wave1data['x'], 'g.', label=wave1)
    plt.plot(wave2data['y'], wave2data['x'], 'b.', label=wave2)
    plt.plot(wave3data['y'], wave3data['x'], 'm.', label=wave3)
    plt.grid()
    plt.legend()
    plt.xlabel('x [pix]')
    plt.ylabel('y [pix]')
    plt.savefig('slits_plot.png')
    plt.show()


def ccf_all():
    rvarr = np.arange(250, 2850, 250)
    for rv in rvarr:
        ccf(rv)



