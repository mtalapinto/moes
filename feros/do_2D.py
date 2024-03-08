import numpy as np
import os
import pandas as pd
from scipy import interpolate
import math
from optics import echelle_orders
from optics import spectrograph
import sys
import matplotlib.pyplot as plt


def do_2D(rv, fcam):
    #moesdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/moes_files/'+str(int(rv))+'/'
    moesdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/moes_files/' + str(int(rv)) + '/'
    outdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/'+str(int(rv))+'/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print('Creating 2D spectra...')
    omin = 63
    omax = 104
    print(rv)
    while omin <= omax:
        fileout = outdir + str(int(omin)) + '_2D_full.tsv'
        print(fcam, rv, omin)
        if not os.path.exists(fileout):
            fileorder = open(fileout, 'w')
            fileorder.write('xpix,ypix,wave,flux\n')
            orddata = pd.read_csv(moesdir + str(int(omin)) + '.tsv', sep=',')
            orddata = orddata.loc[orddata['x'] <= 2048]
            orddata = orddata.loc[orddata['x'] >= 0]
            orddata = orddata.loc[orddata['y'] <= 2048]
            orddata = orddata.loc[orddata['y'] >= 0]
            ymin = np.min(orddata['y'].values) + 1
            ymax = np.max(orddata['y'].values) - 1
            yini = int(ymin)
            yend = int(ymax)

            while yini <= yend:
                data_per_ypix = orddata.loc[orddata['y'] > yini - 0.5]
                data_per_ypix = data_per_ypix.loc[data_per_ypix['y'] <= yini + 0.5]
                xline = np.mean(data_per_ypix['x'].values)
                yline = yini
                waveline = np.mean(data_per_ypix['wave'].values)
                fluxline = np.mean(data_per_ypix['flux'].values)
                fileorder.write('%f,%f,%f,%f\n' % (float(xline), float(yline), float(waveline), float(fluxline)))
                yini += 1
            fileorder.close()

        else:
            print('File already created...')
        omin += 1


def do_2D_order(rv, fcam, omin):
    moesdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/moes_files/'+str(int(rv))+'/'
    outdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/'+str(int(rv))+'/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print('Creating 2D spectra...')
    fileout = outdir + str(int(omin)) + '_2D.tsv'
    print(fcam, rv, omin)
    x, y, waveout, fluxout = [], [], [], []
    if not os.path.exists(fileout):
        #fileorder = open(fileout, 'w')
        #fileorder.write('xpix,ypix,wave,flux\n')
        orddata = pd.read_csv(moesdir + str(int(omin)) + '.tsv', sep=',')
        orddata = orddata.loc[orddata['x'] <= 2048]
        orddata = orddata.loc[orddata['x'] >= 0]
        orddata = orddata.loc[orddata['y'] <= 2048]
        orddata = orddata.loc[orddata['y'] >= 0]
        ymin = np.min(orddata['y'].values) + 1
        ymax = np.max(orddata['y'].values) - 1
        yini = int(ymin)
        yend = int(ymax)

        while yini <= yend:
            data_per_ypix = orddata.loc[orddata['y'] > yini - 0.5]
            data_per_ypix = data_per_ypix.loc[data_per_ypix['y'] <= yini + 0.5]
            xline = np.mean(data_per_ypix['x'].values)
            yline = yini
            waveline = np.mean(data_per_ypix['wave'].values)
            fluxline = np.mean(data_per_ypix['flux'].values)
            x.append(xline)
            y.append(yline)
            waveout.append(waveline)
            fluxout.append(fluxline)
            #fileorder.write('%f,%f,%f,%f\n' % (float(xline), float(yline), float(waveline), float(fluxline)))
            yini += 1
        #fileorder.close()
    else:
        print('File already created...')

    outdata = pd.DataFrame()
    outdata['xpix'] = x
    outdata['ypix'] = y
    outdata['wave'] = waveout
    outdata['flux'] = fluxout
    outdata.to_csv(fileout, index=False)
    return 0


def do_2D_v2(rv, fcam):
    moesdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/moes_files//' + str(int(rv)) + '/'
    outdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/' + str(int(rv)) + '/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print('Creating 2D spectra...')
    omin = 63
    omax = 104
    print(rv)
    while omin <= omax:
        orddata = pd.read_csv(moesdir + str(int(omin)) + '.tsv', sep=',')
        orddata = orddata.loc[orddata['x'] <= 2048]
        orddata = orddata.loc[orddata['x'] >= 0]
        orddata = orddata.loc[orddata['y'] <= 2048]
        orddata = orddata.loc[orddata['y'] >= 0]
        x = []
        y = []
        waveout = []
        flux = []
        waves = np.unique(orddata['wave'])
        print(fcam, rv, omin)
        for wave in waves:
            wavedata = orddata.loc[orddata['wave'] == wave]
            x.append(np.mean(wavedata['x'].values))
            y.append(np.mean(wavedata['y'].values))
            flux.append(np.mean(wavedata['flux'].values))
            waveout.append(wave)

        fileout = outdir + str(int(omin)) + '_2D_v2.tsv'
        if not os.path.exists(fileout):
            dataout = pd.DataFrame()
            dataout['xpix'] = x
            dataout['ypix'] = y
            dataout['wave'] = waveout
            dataout['flux'] = flux
            dataout.to_csv(fileout, index=False)

        else:
            print('File already created...')
        omin += 1


def pixelize_2D(rv, fcam):
    moesdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/moes/' + str(int(rv)) + '/'
    outdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/' + str(int(rv)) + '/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print('Creating 2D spectra...')
    omin = 63
    omax = 104
    print(rv)
    while omin <= omax:
        print(omin)
        orddata = pd.read_csv(outdir + str(int(omin)) + '_2D_v2.tsv', sep=',')
        pixarray = np.arange(0.5, 2047.5, 1.)
        fluxfunc = interpolate.interp1d(orddata['ypix'], orddata['flux'])
        wavefunc = interpolate.interp1d(orddata['ypix'], orddata['wave'])
        ypixfunc = interpolate.interp1d(orddata['ypix'], orddata['ypix'])

        fluxout = fluxfunc(pixarray)
        waveout = wavefunc(pixarray)
        ypixout = ypixfunc(pixarray)
        pixarray = pixarray - 0.5

        '''
        pixmin = int(min(orddata['xpix']))
        pixmax = int(max(orddata['xpix']))

        x, y, wave, flux = [], [], [], []

        while pixmin < pixmax:
            data = orddata.loc[orddata['xpix'] < pixmin + 1]
            data = data.loc[data['xpix'] >= pixmin]

            x.append(pixmin)
            y.append(np.mean(data['ypix']))
            wave.append(np.mean(data['wave']))
            flux.append(np.mean(data['flux']))
            pixmin += 1
        '''
        fileout = outdir + str(int(omin)) + '_2D_v2_red.tsv'
        dataout = pd.DataFrame()
        dataout['xpix'] = pixarray
        dataout['ypix'] = ypixout
        dataout['wave'] = waveout
        dataout['flux'] = fluxout
        dataout.to_csv(fileout, index=False)

        omin += 1


def do_2D_all(fcam):
    rvs = np.arange(3750, 10001, 250)
    for rv in rvs:
        do_2D(rv, fcam)


def create_moes_spectra(rv, o, det, n):
    order = echelle_orders.init_stellar_doppler_simple(rv, o)
    spec = spectrograph.tracing_det(order, det)
    #print(spec['y'])
    print('moes ray tracing... done')
    wmin = min(spec['wave'].values) * 1e4
    wmax = max(spec['wave'].values) * 1e4
    pini = int(min(spec['x'].values))
    pend = int(max(spec['x'].values))
    waux = wmin

    waveout, fluxout, pixout, pixy = [], [], [], []
    while pini <= pend:
        pixdata = spec.loc[spec['x'] >= pini]
        pixdata = pixdata.loc[pixdata['x'] <= pini + 1]

        if math.isnan(np.mean(pixdata['flux'].values)):
            print(pini, pend)
            print(pixdata)
        waveout.append(np.mean(pixdata['wave']))
        fluxout.append(np.mean(pixdata['flux']))

        pixout.append(pini)
        pixy.append(np.mean(np.mean(pixdata['y'])))
        pini += 1

    dataout = pd.DataFrame()
    dataout['wave'] = waveout
    dataout['flux'] = fluxout
    dataout['pix'] = pixout
    dataout['pixy'] = pixy
    print('Saving .tsv file... '),
    outdir = "".join(['data/pix_exp/ns0/ccd_', str(n), '/', str(rv), '/'])
    dataout.to_csv(outdir + str(o) + '_2D_moes.tsv', sep=',', index=False)
    print('done')


def do_all_2D_moes_simple(n):
    x_um = 4096. * 15.0
    y_um = 2048. * 15.0
    #pixarray = np.arange(9, 21.5, 1.5)
    #pixarray = [6.5]
    fcam = 410.
    fcol = 1501.
    slit = 120
    pixarray = np.arange(7.5, 21.1, 1.5)
    basedir = "".join(['data/pix_exp/ns0/'])
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    rvs = np.arange(-10000, 10001, 100)
    #rvs = [-7200]

    print('FEROS MOES spectra creation')
    pixarray = [pixarray[int(n)]]
    for pixsize in pixarray:
        samp = fcam / fcol * slit / pixsize
        x_pix = x_um / pixsize
        y_pix = y_um / pixsize
        det = [samp, pixsize, x_pix, y_pix]
        print('Detector no. ', str(n))
        print('Detector size = ', int(y_pix), 'x', int(x_pix))
        print('Pixel size = ', pixsize)
        print('Sampling = ', np.round(samp, 2))
        detdir = "".join([basedir, 'ccd_', str(int(n)), '/'])
        if not os.path.exists(detdir):
            os.mkdir(detdir)
        for rv in rvs:
            print('RV = ', rv, 'm/s')
            rvoutdir = detdir + str(rv) + '/'
            if not os.path.exists(rvoutdir):
                os.mkdir(rvoutdir)

            omin = 32
            omax = 60
            while omin <= omax:
                create_moes_spectra(rv, omin, det, n)
                omin += 1


def create_all_noisy_data():
    rvs = np.arange(-10000, 10001, 100)
    dets = np.arange(0, 10, 1)
    omax = 60
    ns = 0.1 # 1%
    for det in dets:
        for rv in rvs:
            omin = 32
            while omin <= omax:
                filetemp = 'data/pix_exp/ns'+str(int(ns*100))+'/ccd_'+str(det)+'/'+str(rv)+'/'+str(omin)+'_2D_moes.tsv'
                if not os.path.exists(filetemp):
                    create_noisy_data(det, rv, omin, ns)
                else:
                    print('File already created.')
                omin += 1


def create_noisy_data(det, rv, order, sn):
    print('Creating noisy data')
    print(det, rv, order, sn)

    #stardata = pd.read_csv('stellar_template/stellar_template_v2.tsv', sep=' ')
    #wini = 3800
    #wend = 6800
    basedir = 'data/pix_exp/'
    sndir = basedir+'ns'+str(int(sn*100))+'/'
    if not os.path.exists(sndir):
        os.mkdir(sndir)
    detdir = sndir+'ccd_'+str(det)+'/'
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    rvdir = detdir+str(rv)+'/'
    if not os.path.exists(rvdir):
        os.mkdir(rvdir)

    data = pd.read_csv(basedir+'ns0/ccd_'+str(det)+'/'+str(rv)+'/'+str(order)+'_2D_moes.tsv', sep=',')
    #wini = min(data['wave'].values)
    #wend = max(data['wave'].values)
    #stardata = stardata.loc[stardata['WAVE'] <= wend*1e4]
    #stardata = stardata.loc[stardata['WAVE'] >= wini*1e4]
    #snr_temp = DER_SNR(stardata['FLUX'])
    #print(snr_temp)
    # we add noise of 5%
    #snr_i = DER_SNR(data['flux'].values)
    #print(snr_i)
    ns = np.random.normal(0, 1*sn, len(data))
    data['flux_ns'] = data['flux'] + ns

    dataout = pd.DataFrame()
    dataout['wave'] = data['wave']
    dataout['flux'] = data['flux_ns']
    dataout['pix'] = data['pix']
    dataout['pixy'] = data['pixy']
    dataout.to_csv(rvdir+str(order)+'_2D_moes.tsv', index=False, sep=',')
    #snr_f = DER_SNR(data['flux_ns'].values)
    #print(snr_f)
    #plt.figure(figsize=[8, 3])
    #plt.plot(data['wave'], data['flux'], 'k-', alpha=0.5)
    #plt.plot(data['wave'], data['flux_ns'], 'b-', alpha=0.5)
    #plt.show()
    # signal = np.median(data['flux'].values)
    # noise = np.sqrt(signal)
    # snr_ori = signal/noise
    # signal_new = np.median(data['flux_ns'].values)
    # noise_new = np.sqrt(signal_new)
    # snr_new = signal_new / noise_new
    # print(snr_ori, snr_new)


def detectors_compare():
    x_um = 4096. * 15.0
    y_um = 2048. * 15.0
    # pixarray = np.arange(9, 21.5, 1.5)
    # pixarray = [6.5]
    fcam = 410.
    fcol = 1501.
    slit = 120
    spectrum = echelle_orders.init()
    pixarray = np.arange(7.5, 21.1, 1.5)
    for pixsize in pixarray:
        samp = fcam / fcol * slit / pixsize
        x_pix = x_um / pixsize
        y_pix = y_um / pixsize
        det = [samp, pixsize, x_pix, y_pix]
        print(det)
        ws = spectrograph.tracing_det(spectrum, det)
        plt.clf()
        plt.plot(ws['x'], ws['y'], 'k.')
        plt.xlim(0, x_pix)
        plt.ylim(0, y_pix)
        plt.show()
        plt.clf()


if __name__ == '__main__':

    #do_2D_all(230)
    #do_2D_v2(0, 300)
    create_all_noisy_data()
    #det = [2.02, 15, 4096, 2048]
    #n = 0
    #create_moes_spectra(0, 40, det, n)
    #do_all_2D_moes_simple(sys.argv[1])
    #detectors_compare()
    #

    #do_all_2D_moes_simple(int(sys.argv[1]))
    #do_2D_order(-6750, 230, 95)
    #import time

    #time.sleep(10)
