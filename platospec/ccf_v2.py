import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import argrelextrema


def ccf_new(rv, fcam, instr):
    wmin = 0.38
    wmax = 0.68

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
    zerodir = "".join([basedir, str(instr), '/data/f', str(int(fcam)),'mm/', str(0), '/'])
    indir = "".join([basedir, str(instr), '/data/f', str(int(fcam)), 'mm/', str(0) + '/'])
    stardir = '/home/eduspec/Documentos/moes/stellar_template/'

    #zerodir = "".join(
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
    zerodata = pd.read_csv("".join([zerodir, str(order)+'_2D.tsv']), sep=',')
    zerodata['wave'] = zerodata['wave'] * 1e4
    #print(zerodata)

    wmin0 = min(zerodata['wave'].values)
    wmax0 = max(zerodata['wave'].values)
    plt.plot(zerodata['wave'], zerodata['flux'], 'r-', alpha=0.5)
    plt.plot(temp['wave'], temp['flux'], 'k-', alpha=0.5)
    plt.xlim(wmin0, wmax0)
    #plt.show()
    plt.clf()
    #plt.show()
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
                #stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
                #temp = pd.read_csv(stardir + 'P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec',
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

                #plt.plot(obs['wave'], obs['flux'], 'k-', alpha=0.5)
                #plt.plot(maskord['wini'], np.full(len(maskord), 1), 'ro')
                #plt.plot(maskord['wend'], np.full(len(maskord), 1), 'ro')
                #plt.xlim(6749.75, 6750.50)
                #plt.show()
                #plt.clf()
                ccf_ord = 0.
                wsum_ord = 0.

                for i in range(len(maskord)):
                    #print(mask)
                    mask_weight = maskord['weight'].values[i]
                    pixmin = maskord['pixini'].values[i]
                    pixmax = maskord['pixend'].values[i]

                    obsdata = obs.loc[obs['y'] <= pixmax]
                    obsdata = obsdata.loc[obsdata['y'] >= pixmin]

                    #plt.plot(obs['wave'], obs['flux'], 'k-', alpha=0.5)
                    #plt.plot(maskord['wini'].values[i], 1, 'ro')
                    #plt.plot(maskord['wend'].values[i], 1, 'ro')

                    #plt.xlim(maskord['wini'].values[i] - 0.2, maskord['wend'].values[i] + 0.2)
                    #plt.show()
                    #plt.clf()
                    #print(obsdata)

                    #print(len(obsdata))
                    for k in range(len(obsdata)):
                        minpix_obs = obsdata['y'].values[k] - 0.5
                        maxpix_obs = obsdata['y'].values[k] + 0.5

                        if minpix_obs >= maskord['pixini'].values[i] and maxpix_obs <= maskord['pixend'].values[i]:
                            dpix = 1.
                            #print(dpix, 0)
                            ccfpix = obsdata['flux'].values[k] * mask_weight * dpix
                            ccf_ord += ccfpix
                            wsum_ord += mask_weight**2
                        elif minpix_obs >= maskord['pixini'].values[i] and maxpix_obs >= maskord['pixend'].values[i]:
                            dpix = np.abs(minpix_obs - maskord['pixend'].values[i])
                            #print(dpix, 1)
                            ccfpix = obsdata['flux'].values[k] * mask_weight * dpix
                            ccf_ord += ccfpix
                            wsum_ord += mask_weight**2
                        elif minpix_obs <= maskord['pixini'].values[i] and maxpix_obs <= maskord['pixend'].values[i]:
                            dpix = np.abs(maskord['pixini'].values[i] - maxpix_obs)
                            #print(dpix, 2)
                            ccfpix = obsdata['flux'].values[k] * mask_weight * dpix
                            ccf_ord += ccfpix
                            wsum_ord += mask_weight**2

                        # Check for pixel centers outside the mask
                        if len(obsdata) == 1:
                            if minpix_obs > maskord['pixini'].values[i]:
                                # print(obsdata['y'].values['k'] - 1)
                                outpixdata = obs.loc[obs['y'] == (obsdata['y'].values[k] - 1)]
                                outdpix = np.abs(((outpixdata['y'] + 0.5) - maskord['pixini'].values[i]).values[0])

                                #print(outdpix, mask_weight, outpixdata['flux'].values[0])
                                ccfoutpix = outdpix * mask_weight * outpixdata['flux'].values[0]
                                ccf_ord += ccfoutpix
                                wsum_ord += mask_weight**2
                                #plt.plot(outpixdata['y'], 1, 'go')
                                #plt.plot(outpixdata['y'] + 0.5, 1, 'bo')
                                #plt.plot(outpixdata['y'] - 0.5, 1, 'bo')
                            if maxpix_obs < maskord['pixend'].values[i]:
                                outpixdata = obs.loc[obs['y'] == (obsdata['y'].values[k] + 1)]
                                outdpix = np.abs(((outpixdata['y'] - 0.5) - maskord['pixend'].values[i]).values[0])
                                #print(outdpix, mask_weight, outpixdata['flux'].values[0])
                                ccfoutpix = outdpix * mask_weight * outpixdata['flux'].values[0]
                                ccf_ord += ccfoutpix
                                wsum_ord += mask_weight**2
                                #plt.plot(outpixdata['y'], 1, 'go')
                                #plt.plot(outpixdata['y'] + 0.5, 1, 'bo')
                                #plt.plot(outpixdata['y'] - 0.5, 1, 'bo')

                        #plt.plot(obs['y'], obs['flux'], 'k-', alpha=0.5)
                        #plt.plot(maskord['pixini'].values[i], 1, 'ro')
                        #plt.plot(maskord['pixend'].values[i], 1, 'ro')
                        #plt.plot(minpix_obs, 1, 'bo')
                        #plt.plot(maxpix_obs, 1, 'bo')
                        #plt.plot(obsdata['y'].values[k], 1, 'ko')
                        #plt.xlim(minpix_obs - 2, maxpix_obs + 2)
                        #plt.show()
                        #plt.clf()
                ccf_out += ccf_ord#/np.sqrt(wsum_ord)
                ominaux += 1
            #print(ccf_ord)
            ccf_i.append(ccf_out)
            rv_array.append(float(rvmin))
            #rms_temp = np.sqrt(flux_temp_aux / Ntot)
            #rms_obs = np.sqrt(flux_obs_aux / Ntot)
            #ccf.append(float(ccsum / (Ntot * rms_temp * rms_obs)))
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
        #plt.clf()
    else:
        print('File already created...')

    return 0


def ccf_gauss_fit_v2(rv, fcam):
    #ccfdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(fcam)), 'mm/ccf/'])
    ccfdir = "".join(['/home/eduspec/Documentos/moes/platospec/data/f', str(int(fcam)), 'mm/ccf/'])
    ccf = pd.read_csv(ccfdir + 'ccf_' + str(int(rv)) + '.tsv', sep=',')
    outfile = '/home/eduspec/Documentos/moes/platospec/data/f'+str(int(fcam))+'mm/plots/ccf_gauss_'+str(int(rv))+'.png'
    #ccf = ccf.loc[ccf['ccf_norm'] > ccf['ccf_norm'].values[-1]]
    x = ccf['rv']
    y = ccf['ccf']

    mean = rv*1e-3
    sigma = np.std(ccf['rv'].values)
    height = min(ccf['ccf'])
    offset = 0.1
    print(mean)
    popt, pcov = curve_fit(gaus, ccf['rv'], ccf['ccf'], p0=[height, mean, sigma, offset])

    #g_init = models.Gaussian1D(amplitude=1., mean=rv*1e-3, stddev=3.)
    #fit_g = fitting.LevMarLSQFitter()
    #g = fit_g(g_init, x, y)
    #print()
    #print(g)
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'ko')
    plt.plot(x, gaus(x, *popt), label='Gaussian fit RV mean = '+str(popt[1]*1e3) + ' m/s')
    print(popt[1])
    plt.title('CCF for a nominal RV = '+str(rv)+' m/s')
    plt.xlabel('RV (km/s)   ')
    plt.ylabel('CCF')
    plt.legend(loc='best')
    #plt.savefig(outfile)
    plt.show()

    plt.close()
    rvdiff = rv - popt[1]*1e3
    print(rvdiff)
    return rvdiff
