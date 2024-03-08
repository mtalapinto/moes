from scipy.signal import argrelextrema
import pandas as pd
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from astropy.io import ascii
import numpy as np
from scipy.optimize import curve_fit
import os
import echelle_orders
import platospec_moes
from scipy import interpolate


def convolve_spectrum(rv):
    basedir = 'data/convolved_spectra/'
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    spec = pd.read_csv('stellar_template/stellar_template_resampled.tsv', sep=',')
    spec = spec.loc[spec['wave'] < 7200]
    spec = spec.loc[spec['wave'] > 3600]

    R = 2500
    fwhm = spec['wave'] / R
    std = fwhm / 2.355

    #   Convolve using astropy.convolution
    kernel = Gaussian1DKernel(stddev=np.mean(std))
    convoluted = convolve(spec['flux'].values, kernel, normalize_kernel=True,
                          boundary='extend')

    convspec = pd.DataFrame()

    convspec['wave'] = spec['wave'] * (1 + rv / 3e8)
    convspec['flux'] = convoluted
    convspec.to_csv(basedir+'rv_'+str(rv)+'_conv.tsv', index=False, sep=',')
    #get_res(spec, 'temp')
    #get_res(convspec, 'conv')


def pixelize_convolved_spectrum(rv):
    basedir = 'data/convolved_spectra/'
    convspec = pd.read_csv(basedir+'rv_'+str(rv)+'_conv.tsv', sep=',')
    refdata = pd.read_csv('ref4conv.dat', sep=',')

    pixdir = 'data/pixelized/'
    if not os.path.exists(pixdir):
        os.mkdir(pixdir)

    fcam = 240.
    fcol = 876.
    slit = 100
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(9, 18, 1.5)
    #pixarray = [7.5]
    m = 1
    for pixsize in pixarray:
        samp = fcam / fcol * slit / pixsize
        print('Pixelizing convolved spectra for RV = ', rv, ' and sampling = ', samp)
        x_pix = x_um / pixsize
        y_pix = y_um / pixsize
        npix = int((x_pix + y_pix)/2)
        detdir = pixdir + 'ccd_'+str(m)+'/'
        if not os.path.exists(detdir):
            os.mkdir(detdir)
        outdir = detdir + 'rv_' + str(rv) + '/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        dlambda = np.abs(refdata['wend'].values[0] * 1e4 - refdata['wini'].values[0] * 1e4) / npix  # in A
        for i in range(len(refdata)):
            print('RV = ', rv, ', order = ', refdata['order'].values[i], ', sampling = ', samp)
            order = convspec.loc[convspec['wave'] <= refdata['wend'].values[i]*1e4]
            order = order.loc[order['wave'] >= refdata['wini'].values[i]*1e4]

            wmin = min(order['wave'])
            wmax = max(order['wave'])
            k = 0
            waux = wmin
            #print(wmin)
            waveout, pixout, fluxout, sampout = [], [], [], []
            while waux <= wmax:

                wpixmin = waux - dlambda / 2
                wpixmax = waux + dlambda / 2
                #print('waves', wpixmin, wpixmax)
                pixdata = order.loc[order['wave'] >= wpixmin]
                pixdata = pixdata.loc[pixdata['wave'] <= wpixmax]
                #print(dlambda)
                #print(len(pixdata), k)
                if len(pixdata) > 0:
                    #print(k, npix)
                    flux = np.mean(pixdata['flux'])
                    waveout.append(waux)
                    fluxout.append(flux)
                    pixout.append(k)
                    sampout.append(samp)
                    waux += dlambda

                k += 1

            ordout = pd.DataFrame()
            ordout['wave'] = waveout
            ordout['flux'] = fluxout
            ordout['pix'] = pixout
            ordout['samp'] = sampout
            #ordout.dropna()
            ordout.to_csv(outdir + str(int(refdata['order'].values[i])) + '_2D_pix.tsv', sep=',', index=False)

        m += 1


def gaus(x, height, x0, sigma, offset):
    return height*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset


def get_res(spec):
    spec = spec.sort_values('wave')
    spec['min'] = spec.iloc[argrelextrema(spec.flux.values, np.less_equal, order=3)[0]]['flux']
    #print(spec['min'])
    tempmin = spec.dropna()
    print(tempmin)
    w, wcenout, R, flux_cont, Im = [], [], [], [], []
    plt.plot(spec['wave'], spec['flux'], 'k-', alpha=0.5)
    plt.plot(tempmin['wave'], tempmin['flux'], 'ro')
    plt.show()
    plt.clf()

    for k in range(len(tempmin)):

        wcen = tempmin['wave'].values[k]
        fluxmin = tempmin['flux'].values[k]
        dlambda = 0.07
        linedata = spec.loc[spec['wave'] < wcen + dlambda]
        linedata = linedata.loc[linedata['wave'] > wcen - dlambda]

        linewings_r = linedata.iloc[-3:]
        linewings_l = linedata.iloc[:3]
        slope_wing_r = (linewings_r['flux'].values[-1] - linewings_r['flux'].values[0]) / (
                linewings_r['wave'].values[-1] - linewings_r['wave'].values[0])
        slope_wing_l = (linewings_l['flux'].values[-1] - linewings_l['flux'].values[0]) / (
                linewings_l['wave'].values[-1] - linewings_l['wave'].values[0])

        flux_wing_r = np.mean(linewings_r['flux'])
        std_flux_wing_r = np.std(linewings_r['flux'])
        flux_wing_l = np.mean(linewings_l['flux'])
        std_flux_wing_l = np.std(linewings_l['flux'])
        std_threshold = 0.1
        delta_flux = flux_wing_l - flux_wing_r
        amp_flux = max(linedata['flux']) - min(linedata['flux'])
        if std_flux_wing_r < std_threshold and std_flux_wing_l < std_threshold and np.abs(delta_flux) < 0.1 and len(
                linedata) > 6 and amp_flux > 0.1:
            linedata['min'] = linedata.iloc[argrelextrema(linedata.flux.values, np.less_equal, order=15)[0]][
                'flux']
            linemins = linedata.dropna()
            if 0 < len(linemins) < 2 and min(linemins['wave']) != min(linedata['wave']) and max(
                    linemins['wave']) != max(linedata['wave']):
                # print(flux_wing_r, flux_wing_l, len(linewings_l), slope_wing_l, slope_wing_r, std_flux_wing_r,
                #      std_flux_wing_l)
                x = linedata['wave']
                y = linedata['flux']

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
                        # plt.plot(linedata['wave'], linedata['flux'], 'k-')
                        # plt.plot(linemins['wave'], linemins['flux'], 'bo', markersize=5)
                        # plt.plot(linewings_r['wave'], linewings_r['flux'], 'r-')
                        # plt.plot(linewings_l['wave'], linewings_l['flux'], 'r-')
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


def Rplot():
    temp = pd.read_csv('line_list_temp.tsv', sep=',')
    conv = pd.read_csv('line_list_conv.tsv', sep=',')

    plt.plot(temp['wave'], temp['R'], 'b.', alpha=0.5)
    plt.plot(conv['wave'], conv['R'], 'r.', alpha=0.5)
    plt.show()
    Rconv = np.mean(conv['R'])
    print(Rconv)


def convplot():
    temp = pd.read_csv('stellar_template/stellar_template.tsv',sep=',')
    conv = pd.read_csv('stellar_template/stellar_template_conv.tsv', sep=',')
    order = 80
    pixd = pd.read_csv('data/reference_spectra/rv_0/'+str(int(order))+'_2D_conv.tsv', sep=',')

    plt.plot(temp['wave'], temp['flux'], 'k-', alpha=0.5)
    plt.plot(conv['wave'], conv['flux'], 'r-', alpha=0.5)
    plt.plot(pixd['wave'], pixd['flux'], 'b-', alpha=0.8)
    plt.show()


def conv_all():
    rvs = np.arange(-10000, 10001, 50)
    for rv in rvs:
        print('Creating convolved spectra for RV = ', rv, ' m/s')
        convolve_spectrum(rv)


def pix_all():
    rvs = np.arange(-10000, 10001, 50)
    #rvs = [0]
    for rv in rvs:
        print('Creating convolved spectra for RV = ', rv, ' m/s')
        pixelize_convolved_spectrum(rv)


def do_ref():
    omin = 68
    omax = 122

    spec = echelle_orders.init()
    det = [2.02, 13.5, 2048, 2048, 4]
    specout = platospec_moes.tracing_full_det(spec, det)
    specaux = specout.loc[specout['x'] >= -1]
    specaux = specaux.loc[specaux['x'] <= 2049]
    specaux = specaux.loc[specaux['y'] <= 2048]
    specaux = specaux.loc[specaux['y'] >= 0]
    orders = np.unique(specaux['order'].values)
    print(orders)
    orders = orders[:-1]
    #orders = orders[1:]
    specaux = specaux.loc[specaux['order'] < max(orders)]
    specaux = specaux.loc[specaux['order'] > min(orders)]
    orders = np.unique(specaux['order'].values)
    #plt.plot(specaux['x'], specaux['y'], 'k.')
    ordout, wminout, wmaxout = [], [], []
    for order in orders:
        orderdat = specaux.loc[specaux['order'] == order]
        wmin = min(orderdat['wave'].values)
        wmax = max(orderdat['wave'].values)
        wminout.append(wmin)
        wmaxout.append(wmax)
        ordout.append(order)

    outdata = pd.DataFrame()
    outdata['order'] = ordout
    outdata['wini'] = wminout
    outdata['wend'] = wmaxout
    outdata.to_csv('ref4conv.dat', sep=',', index=False)


def resample_template():
    temp = 'stellar_template/stellar_template.tsv'
    data = pd.read_csv(temp, sep=',')
    resampspec = interpolate.interp1d(data['wave'], data['flux'])
    data = data.iloc[:-10,:]
    data = data.iloc[10:, :]
    wmin = min(data['wave'].values)
    wmax = max(data['wave'].values)
    res = np.mean(data.diff()['wave'].dropna().values)
    wrange = np.arange(wmin, wmax, res/4)
    newflux = resampspec(wrange)
    specout = pd.DataFrame()
    specout['wave'] = wrange
    specout['flux'] = newflux
    specout.to_csv('stellar_template/stellar_template_V3.tsv', sep=',', index=False)
    print('template spectrum resampled.')



if __name__ == '__main__':

    #convolve_spectrum()
    #Rplot()
    #conv_all()
    #pix_all()
    #convolve_spectrum(0)
    #conv_all()
    #do_ref()
    #convolve_spectrum(0)
    #pix_all()
    resample_template()
    #pixelize_convolved_spectrum(0)
    #resample_template()
    #convplot()

