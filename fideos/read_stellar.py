from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.constants import h, k, c


def phoenix_test():
    stellarpath = 'C:\\Users\\marce\\Documents\\fondecyt_uai\\fideos_moes\\stellar_spectrum\\'
    filename = 'lte05700-3.00+1.0.7.dat'

    #hdu = open(stellarpath+filename,'r')
    data = pd.read_csv(stellarpath+filename, sep=',')

    wavemin = 3500.
    wavemax = 9000.
    data = data.loc[data['wave'].values < wavemax]
    data = data.loc[data['wave'].values > wavemin]
    waves = data['wave'].values
    #print(data.diff())
    #print(3500/0.048)
    #print(9000/0.080)
    plt.figure(figsize=(12, 4))
    plt.plot(data['wave'],data['flux'],'k.')
    plt.show()
    #data['wave'] = data['wave'].values[:].replace('D', 'E')

    #data = data.loc[data['wave'].values.astype(np.float) > wavemin]
    #data = data.loc[data['wave'].values.astype(np.float) < wavemax]
    #fout = open(stellarpath+'lte05700-3.00+1.0.7.dat', 'w')
    #fout.write('wave,flux\n')
    #for k in range(len(data)):
    #    wave = float(data['wave'].values[k].replace('D', 'E'))
    #    flux = float(data['flux'].values[k].replace('D', 'E'))
    #    fout.write('%.10f,%.10f\n' %(wave, flux))

    #fout.close()


def spec2D():
    stardata = pd.read_csv('stellar_spectrum/moes_stellar_spectrum_rv_500ms.csv', sep=',')
    stardata = stardata.loc[stardata['x'] <= 2048]
    stardata = stardata.loc[stardata['x'] >= 0]
    stardata = stardata.loc[stardata['y'] <= 2048]
    stardata = stardata.loc[stardata['y'] >= 0]

    orders = np.unique(stardata['order'].values)
    omin = min(orders)
    omax = max(orders) - 1
    while omin <= omax:
        print(omin)
        data = stardata.loc[stardata['order'].values == omin]

        ymin = np.min(data['y'].values) + 1
        ymax = np.max(data['y'].values) - 1
        yini = int(ymin)
        yend = int(ymax)
        fileorder = open('stellar_spectrum/star_shifted_orders/'+str(int(omin))+'_shift_spec.dat', 'w')
        fileorder.write('xpix,ypix,wave,flux\n')
        while yini <= yend:
            data_per_ypix = data.loc[data['y'] > yini - 0.5]
            data_per_ypix = data_per_ypix.loc[data_per_ypix['y'] <= yini + 0.5]
            xline = np.mean(data_per_ypix['x'].values)
            yline = yini
            waveline = np.mean(data_per_ypix['wave'].values)
            fluxline = np.mean(data_per_ypix['flux'].values)
            fileorder.write('%f,%f,%f,%f\n' % (float(xline), float(yline), float(waveline), float(fluxline)))
            yini += 1
        omin += 1
        fileorder.close()

    return 0


def plot_orders():
    stellarpath = 'C:\\Users\\marce\\Documents\\fondecyt_uai\\fideos_moes\\stellar_spectrum\\'
    filename = 'stellar_spectrum_norm.dat'
    origdata = pd.read_csv('stellar_spectrum/' + filename, sep=',')
    omin = 63
    omax = 104
    while omin <= omax:
        data = pd.read_csv('stellar_spectrum/star_template_orders/'+str(omin)+'_star_spec.dat', sep=',')
        data['wave'] = data['wave']*1e4
        wavemin = min(data['wave'].values)
        wavemax = max(data['wave'].values)
        orddata = origdata.loc[origdata['wave'].values < wavemax]
        orddata = orddata.loc[orddata['wave'].values > wavemin]
        orddata = orddata.sort_values(['wave'], ascending=True)
        plt.clf()
        plt.figure(figsize=(16, 4))
        plt.plot(data['wave'], data['flux'], 'r-', zorder=1)
        plt.plot(orddata['wave'], orddata['flux_norm'],'k-',markersize=3, alpha=0.3, zorder=0)
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel('Normalized flux')
        plt.tight_layout()
        plt.savefig('stellar_spectrum/star_template_orders/'+str(omin)+'_fideos_plot.png')
        plt.close()
        omin += 1


def ccf():
    rvmin = -20.0
    rvmax = 20.0
    drv = 0.1
    ccf = []
    rv_array = []
    tempath = 'stellar_spectrum/star_template_orders/'
    shiftpath = 'stellar_spectrum/star_shifted_orders/'
    while rvmin < rvmax:
        print(rvmin)
        ccsum = 0.
        Ntot = 0.
        flux_obs_aux = 0.
        flux_temp_aux = 0.
        omin = 63
        omax = 104
        while omin <= omax:
            ord = omin
            data_template = pd.read_csv(tempath + str(ord) + '_star_spec.dat', sep=',')
            data_shift = pd.read_csv(shiftpath + str(ord) + '_shift_spec.dat', sep=',')
            hirestemp = pd.read_csv('stellar_spectrum/stellar_spectrum_norm.dat', sep=',')

            spectemp = data_template.copy()
            specobs = data_shift.copy()
            specobs['wave'] = specobs['wave'] * 1e4

            hirestemp = hirestemp.loc[hirestemp['wave'] < max(spectemp['wave'] * 1e4)]
            hirestemp = hirestemp.loc[hirestemp['wave'] > min(spectemp['wave'] * 1e4)]

            # Setting wavelength range
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
            cc = tempfunc(obsdata['wave']) * obsdata['flux']
            N = len(cc)
            flux_temp_aux += np.sum(tempfunc(obsdata['wave']) ** 2)
            flux_obs_aux += np.sum(obsdata['flux'].values ** 2)
            #rms_temp = np.sqrt(np.sum(tempfunc(obsdata['wave']) ** 2) / N)
            #rms_obs = np.sqrt(np.sum(obsdata['flux'].values ** 2) / N)
            ccsum += np.sum(cc)
            Ntot += N
            omin += 1

        #print(Ntot)
        #print(flux_obs_aux)
        #print(flux_temp_aux)
        rms_temp = np.sqrt(flux_temp_aux / Ntot)
        rms_obs = np.sqrt(flux_obs_aux / Ntot)
        #print(rms_temp, rms_obs)
        ccf.append(float(ccsum / (Ntot * rms_temp * rms_obs)))
        rv_array.append(float(rvmin))
        rvmin += drv

    plt.clf()
    ccf_data = pd.DataFrame()
    ccf_data['rv'] = rv_array
    ccf_data['ccf'] = ccf
    ccf_data.to_csv('ccf3.dat', index=False)
    plt.plot(rv_array, ccf, 'k.')
    plt.savefig('ccf_test_3.png')
    plt.show()
    plt.clf()
    print(rv_array[np.argmax(ccf)])

    return 0


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


def gaus(x, height ,x0, sigma):
    return height*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def ccf_gaussian_fit():
    ccf = pd.read_csv('ccf3.dat', sep=',')
    ccf = ccf.loc[ccf['rv'] < 12.]
    ccf = ccf.loc[ccf['rv'] > -12.]
    ccf = ccf.loc[ccf['ccf'] > ccf['ccf'].values[-1]]
    ccf['ccf_norm'] = (ccf['ccf'] - min(ccf['ccf'].values))/(max(ccf['ccf'].values) - min(ccf['ccf'].values))

    ccf_func = interpolate.interp1d(ccf['rv'], ccf['ccf_norm'])
    rvarr = np.arange(-10.,10., 0.001)

    ccfarr = ccf_func(rvarr)
    print(ccfarr)
    print(rvarr[np.argmax(ccfarr)])
    mean = np.mean(ccf['rv'].values)
    sigma = np.std(ccf['rv'].values)
    height = max(ccf['ccf'])
    # print(linearr)
    popt, pcov = curve_fit(gaus, ccf['rv'], ccf['ccf_norm'], p0=[height, mean, sigma])
    perr = np.sqrt(np.diag(pcov))
    print(perr)
    plt.plot(ccf['rv'], ccf['ccf_norm'],'k-', label='CCF')
    plt.plot(ccf['rv'], gaus(ccf['rv'], *popt), 'r-', label=r'Gaussian fit, RV$_{mean}$ = '+str(np.round(popt[1]*1.e3, 2))+'+/- ' + str(np.round(perr[1]*1e3, 2)) + 'm/s')
    plt.legend()
    plt.ylabel('CCF')
    plt.xlabel('RV (km/s)')
    plt.tight_layout()
    plt.show()
    #print(popt)
    #print(pcov)


if __name__ == '__main__':

    '''
    stellarpath = 'C:\\Users\\marce\\Documents\\fondecyt_uai\\fideos_moes\\stellar_spectrum\\'
    filename = 'lte05700-3.00+1.0.7.dat'
    origdata = pd.read_csv('stellar_spectrum/'+filename, sep=',')
    print(origdata)
    plt.plot(x0, y0, 'k-', zorder=1)
    plt.plot(x1, y1,'r-', zorder=0)
    plt.show()
    '''
    #ccf()
    ccf_gaussian_fit()

    #spec2D()
    #plot_orders()
    #plot_slits()
    #extract_spectra()
    #spec2D()
    #plot_orders()
    # continuum_fit()

