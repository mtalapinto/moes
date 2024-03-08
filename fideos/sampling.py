import glob
import os.path
import sys

import echelle_orders
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import fideos_spectrograph


def get_sum(vec):
    fvec = np.sort(vec)
    fval = np.median(fvec)
    nn = int(np.around(len(fvec) * 0.15865))
    vali, valf = fval - fvec[nn], fvec[-nn] - fval
    return fval, vali, valf


def gaus(x, height, x0, sigma, offset):
    return height*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset


def get_samp():
    G = 41.6 * 1e-3
    d = 1 / G
    omin = 129
    #omin = 80
    omax = 63
    blaze_angle = 76 * np.pi / 180
    fcam = 135
    fcol = 876
    pixsize = 13.5
    slit = 0.1

    while omin > omax:
        blaze_wave = 2 * np.sin(blaze_angle) / (G * omin)

        ad = omin / (d * np.cos(blaze_wave))
        ld = ad * fcam
        R = 2 * np.tan(blaze_angle) * fcol / slit

        print(blaze_wave, omin, R)
        dlambda = blaze_wave / R
        res_element = ld * dlambda
        s_ccd = fcam/fcol * slit
        #print(s_ccd * 1e3 / pixsize)
        dl = blaze_wave * fcam * omin / (R * np.cos(blaze_angle) * d)
        dpix = dl*1e3/pixsize
        #print(dpix)
        samp = 2.02
        drv = 3e5/(R * samp)
        print(drv)


        omin -= 1


def do_sampling_files():
    rv = 0
    instr = 'platospec'
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(4.5, 21.1, 1.5)
    # pixarray = [6.5]
    fcam = 280.
    fcol = 876.
    slit = 100
    ns = 0


    basedir = '/luthien/platospec/data/pix_exp2/ns'+str(ns)+'/'
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    n = 10
    rv = 0
    for k in range(n):
        omin = 68
        omax = 122
        print(k)
        detdir = basedir + 'ccd_' + str(k) + '/' + str(rv) + '/'

        w, wcenout, samp, flux_cont, Im = [], [], [], [], []
        ordout = []
        sampout = []

        while omin <= omax:
            spec = pd.read_csv(detdir + str(omin) + '_2D_moes.tsv', sep=',')
            spec = spec.sort_values('wave')
            spec['min'] = spec.iloc[argrelextrema(spec.flux.values, np.less_equal, order=5)[0]]['flux']
            tempmin = spec.dropna()
            tempmin = tempmin.loc[tempmin['flux'] < 0.992]
            print(k, omin)
            #plt.plot(spec['pix'], spec['flux'], 'k-', alpha=0.5)
            #plt.plot(tempmin['pix'], tempmin['flux'], 'ro')
            #plt.show()
            #plt.clf()
            #plt.close()
            for i in range(len(tempmin)):

                wcen = tempmin['wave'].values[i]
                pixcen = tempmin['pix'].values[i]
                fluxmin = tempmin['flux'].values[i]
                #print(wcen, fluxmin)
                dlambda = 0.02*1e-3
                linedata = spec.loc[spec['wave'] < wcen + dlambda]
                linedata = linedata.loc[linedata['wave'] > wcen - dlambda]
                #plt.plot(linedata['pix'], linedata['flux'], 'k-')
                #plt.show()
                x = linedata['pix']
                y = linedata['flux']

                # g_init = models.Gaussian1D(amplitude=-1., mean=wcen, stddev=dlambda)
                # fit_g = fitting.LevMarLSQFitter()
                # g = fit_g(g_init, x, y)
                height = min(y)
                mean = pixcen
                sigma = 2.6
                offset = 1.
                try:
                    popt, pcov = curve_fit(gaus, x, y, p0=[height, mean, sigma, offset])
                    perr = np.sqrt(np.diag(pcov))
                    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                    amp = popt[0]
                    pixmean = popt[1]
                    pixdiff = pixmean - pixcen
                    offset = popt[3]
                    w.append(wcen)
                    wcenout.append(pixmean)
                    samp.append(fwhm)
                    flux_cont.append(min(y))
                    Im.append(popt[0] + popt[3])
                    ordout.append(omin)

                except RuntimeError:
                    print('Error in curve fitting...')

            omin += 1
        linelist = pd.DataFrame()
        linelist['wave'] = w
        linelist['pix'] = wcenout
        linelist['s'] = samp
        linelist['order'] = ordout
        outdir = 'stellar_template/sampling/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        linelist.to_csv(outdir + 'ccd_'+str(k)+'_sampling_v2.csv', index=False, sep=',')
    #omin = 71
    #
    #
    #    slit = echelle_orders.init_thar_doppler_full(rv, instr, omin)
    #    waves = np.unique(slit['waves'].values)

    #    for wave in waves:


def do_sampling_files_ccf_lines(instr):
    rv = 0
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    # pixarray = [6.5]
    ns = 0

    if instr == 'platospec':

        fcam = 280.
        fcol = 876.
        slit = 100
        basedir = '/luthien/platospec/data/pix_exp2/ns' + str(ns) + '/'
        omin = 71
        omax = 122
        pixarray = np.arange(4.5, 21.1, 1.5)
        n = len(pixarray)

    elif instr == 'carmenes':
        basedir = '/luthien/carmenes/pix_exp/ns' + str(ns) + '/'
        omin = 87
        omax = 171
        pixarray = np.arange(4.5, 25.6, 1.5)
        n = len(pixarray)

    elif instr == 'feros':
        basedir = '/luthien/feros/data/pix_exp/ns' + str(ns) + '/'
        omin = 32
        omax = 60
        pixarray = np.arange(4.5, 21.1, 1.5)
        n = len(pixarray)

    elif instr == 'fideos':
        basedir = '/luthien/fideos/data/ns' + str(ns) + '/'
        omin = 64
        omax = 106
        pixarray = np.arange(4.5, 24.1, 1.5)
        n = len(pixarray)


    rv = 0
    outdir = '/home/marcelo/Documentos/moes/sampling_plots/sampling_maps/data/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for k in range(n):
        ordmin = omin
        ordmax = omax

        detdir = basedir + 'ccd_' + str(k) + '/' + str(rv) + '/'

        w, wcenout, samp, flux_cont, Im = [], [], [], [], []
        ordout = []
        sampout = []
        mask = pd.read_csv('stellar_template/g2mask_new.tsv', sep=',', names=['wmin', 'wmax'])
        mask['wmin'] = mask['wmin'] * 1e-4
        mask['wmax'] = mask['wmax'] * 1e-4
        if not os.path.exists(outdir + 'ccd_'+str(k)+'_sampling_ccf_lines.csv'):
            while ordmin <= ordmax:
                print(instr, k, ordmin)
                spec = pd.read_csv(detdir + str(ordmin) + '_2D_moes.tsv', sep=',')
                spec = spec.sort_values('wave')
                for l in range(len(mask)):
                    line = spec.loc[spec['wave'].astype(float) < mask['wmax'].values[l].astype(float)]
                    line = line.loc[line['wave'].astype(float) > mask['wmin'].values[l].astype(float)]
                    # print(mask['wmin'].values[l], min(spec['wave']))
                    # print(mask['wmax'].values[l], max(spec['wave']))
                    if len(line) != 0:

                        flux = min(line['flux'])
                        wave = line.loc[line['flux'] == flux]
                        wavemin = wave['wave'].values[0]
                        pixcen = wave['pix'].values[0]
                        lineaux = spec.loc[spec['wave'] < wavemin + 1.5e-5]
                        lineaux = lineaux.loc[lineaux['wave'] > wavemin - 1.5e-5]
                        if len(lineaux) > 3:
                            #plt.plot(lineaux['pix'], lineaux['flux'], 'k-')
                            #plt.show()
                            x = lineaux['pix']
                            y = lineaux['flux']

                            height = min(y)
                            mean = np.median(x)
                            sigma = 2.6
                            offset = 1.
                            try:
                                popt, pcov = curve_fit(gaus, x, y, p0=[height, mean, sigma, offset])
                                perr = np.sqrt(np.diag(pcov))
                                fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                                amp = popt[0]
                                pixmean = popt[1]
                                pixdiff = pixmean - pixcen
                                offset = popt[3]
                                w.append(wavemin)
                                wcenout.append(pixmean)
                                samp.append(fwhm)
                                flux_cont.append(min(y))
                                Im.append(popt[0] + popt[3])
                                ordout.append(ordmin)

                            except RuntimeError:
                                print('Error in curve fitting...')

                ordmin += 1
            linelist = pd.DataFrame()
            linelist['wave'] = w
            linelist['pix'] = wcenout
            linelist['s'] = samp
            linelist['order'] = ordout
            linelist.to_csv(outdir + instr +'_ccd_' + str(k) + '_sampling_ccf_lines_v2.csv', index=False, sep=',')
        else:
            print('File already created')
    #omin = 71
    #
    #
    #    slit = echelle_orders.init_thar_doppler_full(rv, instr, omin)
    #    waves = np.unique(slit['waves'].values)

    #    for wave in waves:


def do_sampling_plots(instr):

    #files = glob.glob('stellar_template/sampling/*')
    ns = 0
    rv = 0
    basedir = '/home/marcelo/Documentos/moes/sampling_plots/sampling_maps/data/'
    if instr == 'platospec':

        fcam = 240.
        fcol = 876.
        slit = 100

        omin = 68
        omax = 122
        pixarray = np.arange(4.5, 19.6, 1.5)
        n = len(pixarray)

    elif instr == 'carmenes':
        fcam = 455.
        fcol = 1590.
        slit = 145.7
        omin = 87
        omax = 171
        pixarray = np.arange(4.5, 24.1, 1.5)
        n = len(pixarray)

    elif instr == 'feros':
        omin = 32
        omax = 60
        fcam = 410.
        fcol = 1501.
        slit = 120
        pixarray = np.arange(4.5, 19.6, 1.5)
        n = len(pixarray)

    elif instr == 'fideos':
        omin = 64
        omax = 106
        fcam = 300.
        fcol = 762.
        slit = 100
        pixarray = np.arange(4.5, 22.6, 1.5)
        n = len(pixarray)

    i = 0
    outdir = '/home/marcelo/Documentos/moes/sampling_plots/sampling_maps/plots/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    while i < n:
        #data = pd.read_csv(basedir + 'ccd_'+str(i)+'_sampling.csv', sep=',')
        dataccf = pd.read_csv(basedir + instr + '_ccd_' + str(i) + '_sampling_ccf_lines.csv', sep=',')
        #data = data.loc[data['s'] < 10.]
        #data = data.loc[data['s'] > 0.9]
        #data = data.loc[data['s'] < np.median(data['s']) + 3 * np.std(data['s'])]
        dataccf = dataccf.loc[dataccf['s'] < 10.]
        dataccf = dataccf.loc[dataccf['s'] > 0.9]
        dataccf = dataccf.loc[dataccf['s'] < np.median(dataccf['s']) + 3 * np.std(dataccf['s'])]
        #data = data.loc[data['s'] > np.median(data['s']) - 3 * np.std(data['s'])]
        imsize = fcam / fcol * slit
        nomsamp = np.round(imsize / pixarray[i], 2)
        #samp, samp_lo, samp_up = get_sum(data['s'].values)
        sampccf, sampccf_lo, sampccf_up = get_sum(dataccf['s'].values)
        plt.figure(figsize=[10, 6])
        #plt.plot(data['order'], data['s'], 'k.', label=' + str(pixarray[i]) + r' $\mu$m' + '\nmean sampling = ' + str(np.round(samp,2)) + ' pix\n nominal sampling = ' + str(nomsamp) + ' pix', alpha=0.5)
        plt.plot(dataccf['order'], dataccf['s'], 'ro',
                 label='Pixel size = ' + str(pixarray[i]) + r' $\mu$m' + '\nmean sampling = ' + str(np.round(sampccf, 2)) + '\n nominal sampling = ' + str(nomsamp) + ' pix', alpha=0.5)
        plt.legend(loc='best')
        plt.xlabel('Spectral order')
        plt.ylabel('Spectral sampling (pix)')
        plt.savefig(outdir + instr + '_ccd_'+str(i) + '.png')
        #plt.show()
        plt.clf()
        plt.close()
        i += 1


def number_of_ccf_lines():
    ns = 0
    basedir = '/luthien/platospec/data/pix_exp2/ns' + str(ns) + '/'
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    outdir = 'stellar_template/sampling/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    mask = pd.read_csv('stellar_template/g2mask_new.tsv', sep=',', names=['wmin', 'wmax'])
    mask['wmin'] = mask['wmin'] * 1e-4
    mask['wmax'] = mask['wmax'] * 1e-4
    n = 10
    i = 0
    fcam = 240  # 240
    fcol = 876
    slit = 100
    rv = 0
    pixarray = np.arange(4.5, 21.1, 1.5)

    for k in range(n):
        nlines = 0
        omin = 71
        omax = 122
        detdir = basedir + 'ccd_' + str(k) + '/' + str(rv) + '/'
        while omin <= omax:
            #print(omin)
            spec = pd.read_csv(detdir + str(omin) + '_2D_moes.tsv', sep=',')
            spec = spec.sort_values('wave')
            for l in range(len(mask)):
                line = spec.loc[spec['wave'].astype(float) < mask['wmax'].values[l].astype(float)]
                line = line.loc[line['wave'].astype(float) > mask['wmin'].values[l].astype(float)]
                # print(mask['wmin'].values[l], min(spec['wave']))
                # print(mask['wmax'].values[l], max(spec['wave']))
                if len(line) != 0:
                    nlines += 1
            omin += 1
        print('Detector ', k, ', pixsize = ', pixarray[k], 'number of CCF lines = ', nlines)


def platospec_compare_plot():
    data0 = pd.read_csv('/home/marcelo/Documentos/moes/sampling_plots/sampling_maps/data/platospec_ccd_6_sampling_ccf_lines_v2.csv', sep=',')
    data1 = pd.read_csv(
        '/home/marcelo/Documentos/moes/sampling_plots/sampling_maps/data/platospec_ccd_6_sampling_ccf_lines.csv',
        sep=',')

    data0 = data0.loc[data0['s'] < 10.]
    data0 = data0.loc[data0['s'] > 0.9]
    data0 = data0.loc[data0['s'] < np.median(data0['s']) + 3 * np.std(data0['s'])]
    data1 = data1.loc[data1['s'] < 10.]
    data1 = data1.loc[data1['s'] > 0.9]
    data1 = data1.loc[data1['s'] < np.median(data1['s']) + 3 * np.std(data1['s'])]

    plt.figure(figsize=[10, 6])
    plt.plot(data1['order'], data1['s'], 'ko', label=r'f$_{cam}$ = 240mm', alpha=0.5)
    plt.plot(data0['order'], data0['s'], 'ro', label=r'f$_{cam}$ = 280mm', alpha=0.5)
    plt.xlabel('Spectral order')
    plt.ylabel('sampling (pix)')
    plt.legend(loc='best')
    plt.savefig('sampling_comparison_cameras.png')
    plt.show()


def inter_order_spacing():
    spectrum = echelle_orders.init_slit()
    # print(spectrum)
    import platospec_moes
    fcam = 240
    fcol = 876.

    slit = 100
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixsize = 13.5
    samp = fcam / fcol * slit / pixsize
    x_pix = x_um / pixsize
    y_pix = y_um / pixsize
    i = 0
    det = [samp, pixsize, x_pix, y_pix, i]
    specout = platospec_moes.tracing_full_fcam(spectrum, fcam)
    specout = specout.loc[specout['x'] >= 0]
    specout = specout.loc[specout['x'] <= 2048]
    specout = specout.loc[specout['y'] >= 0]
    specout = specout.loc[specout['y'] <= 2048]
    # specout = specout.loc[specout['order'] == 73.]
    orders = np.unique(specout['order'])
    minorder = min(orders)
    maxorder = max(orders)
    wmin = min(specout['wave'].values)
    wmax = max(specout['wave'].values)

    omindata = specout.loc[specout['order'] == minorder]
    ominplusonedata = specout.loc[specout['order'] == (minorder + 1)]

    omaxdata = specout.loc[specout['order'] == maxorder]
    omaxminusonedata = specout.loc[specout['order'] == (maxorder - 1)]

    vals = omindata.loc[omindata['x'] < 1024 + 1]
    vals = vals.loc[vals['x'] > 1024]
    minyomin = min(vals['y'])
    ordthick = np.abs(max(vals['y']) - min(vals['y']))
    print('PLATOSpec, fcam = ', fcam, 'mm')
    print('order', minorder)
    print('Order thickness = ', ordthick)
    print('Half-moon thickness = ', ordthick/2)
    vals = ominplusonedata.loc[ominplusonedata['x'] < 1024 + 1]
    vals = vals.loc[vals['x'] > 1024]
    maxyomin = max(vals['y'])

    ios_lo = np.abs(maxyomin - minyomin)
    print('Inter order spacing = ', ios_lo)

    vals = omaxdata.loc[omaxdata['x'] < 1024 + 1]
    vals = vals.loc[vals['x'] > 1024]
    maxyomax = max(vals['y'])
    ordthick = np.abs(max(vals['y']) - min(vals['y']))

    print('order', maxorder)
    print('Order thickness = ', ordthick)
    print('Half-moon thickness = ', ordthick / 2)
    vals = omaxminusonedata.loc[omaxminusonedata['x'] < 1024 + 1]
    vals = vals.loc[vals['x'] > 1024]
    maxyomaxminusone = min(vals['y'])

    ios_hi = np.abs(maxyomax - maxyomaxminusone)
    print('Inter order spacing = ', ios_hi)

    basedir = '/luthien/platospec/data/pix_exp_fcam' + str(fcam) + '/ns0/0/'


def create_sampling_maps(fcam):
    rv = 0
    instr = 'platospec'
    omin = 68
    omax = 122
    #det = []
    fcol = 876.
    slit = 100
    pixsize = 13.5
    x_pix = 2048
    y_pix = 2048
    i = 6
    samp = fcam / fcol * slit / pixsize

    det = [samp, pixsize, x_pix, y_pix, i]
    rv = 0
    waveout, sampout, ordout = [], [], []
    while omin <= omax:
        slitout = echelle_orders.init_points_doppler_full(rv, instr, omin)
        specout = platospec_moes.tracing_full_fcam(slitout, fcam)
        specout = specout.loc[specout['x'] < 2048]
        specout = specout.loc[specout['y'] < 2048]
        specout = specout.loc[specout['x'] > 0]
        specout = specout.loc[specout['y'] > 0]

        waves = np.unique(specout['wave'])

        for wave in waves:
            data = specout.loc[specout['wave'] == wave]
            if len(data) == 2:
                samp = np.abs(data['x'].values[1] - data['x'].values[0])
                #print(data)
                waveout.append(wave)
                sampout.append(samp)
                ordout.append(omin)

        omin += 1

    #plt.figure(figsize=[12, 7])
    #plt.plot(waveout, sampout, '.', label=r'f$_{cam}$ = ' + str(fcam) + ' mm')
    #plt.xlabel('Spectral order')
    #plt.ylabel('Spectral sampling')
    #plt.savefig('results/sampling_map_fcam_'+str(int(fcam))+'.png')
    #plt.show()
    sampling = np.median(sampout)
    print('Mean sampling = ', sampling)
    return sampling



def sampling_maps_det(instr, i):
    rv = 0
    if instr == 'platospec':
        omin = 68
        omax = 122
    elif instr == 'fideos':
        omin = 64
        omax = 106
    elif instr == 'feros':
        omin = 32
        omax = 60
    elif instr == 'carmenes':
        omin = 87
        omax = 171

    #det = []
    x_um = 2048 * 15.0
    y_um = 2048 * 15.0
    #pixarray = np.arange(9, 21.5, 1.5)
    #pixarray = [6.5]
    fcam = 300.
    fcol = 762.
    slit = 100

    imccd = slit * fcam / fcol
    rv = 0
    idet = pd.read_csv('/data/matala/luthien/'+str(instr)+'/data/ns0/ccd_'+str(int(i))+'/0/'+str(int(omin))+'_2D_moes.tsv', sep=',')
    npix = max(idet['pix'])
    
    pixsize = x_um / npix

    samp = fcam / fcol * slit / pixsize
    det = [samp, pixsize, npix, npix, i]
    sampori = imccd / pixsize
    waveout, sampout, ordout = [], [], []
    x, y = [], []
    while omin <= omax:
        slitout = echelle_orders.init_points_doppler_full(rv, instr, omin)
        specout = fideos_spectrograph.raytrace(slitout, det)
        specout = specout.loc[specout['x'] < npix]
        specout = specout.loc[specout['y'] < npix]
        specout = specout.loc[specout['x'] > 0]
        specout = specout.loc[specout['y'] > 0]

        waves = np.unique(specout['wave'])

        for wave in waves:
            data = specout.loc[specout['wave'] == wave]
            if len(data) == 2:
                samp = np.abs(data['x'].values[1] - data['x'].values[0])
                x.append(np.mean(data['x'].values))
                y.append(np.mean(data['y'].values))
                #print(data)
                waveout.append(wave)
                sampout.append(samp)
                ordout.append(omin)

        omin += 1

    sampling = np.median(sampout)
    print('Mean sampling = ', sampling)
    plt.plot(waveout, sampout, '.', label=r's = ' + str(np.round(sampling, 2)) + ' pix, Nominal s = ' + str(np.round(sampori,2)) + ' pix')
    xarr = [min(waveout), max(waveout)]
    yarr = [sampori, sampori]
    plt.plot(xarr, yarr, 'k--')
    yarr = [sampling, sampling]
    plt.plot(xarr, yarr, 'r--')
    #plt.figure(figsize=[12, 7])
    #plt.plot(waveout, sampout, '.', label=r'f$_{cam}$ = ' + str(fcam) + ' mm')
    #plt.xlabel('Spectral order')
    #plt.ylabel('Spectral sampling')
    #plt.savefig('results/sampling_map_fcam_'+str(int(fcam))+'.png')
    #plt.show()

    #plt.close()
    #plt.clf()
    #plt.plot(x,y,'ko')
    #plt.show()
    
    return sampling


if __name__ == '__main__':

    # do_sampling_files()
    #do_sampling_files_ccf_lines(sys.argv[-1])
    #do_sampling_plots(sys.argv[-1])
    #inter_order_spacing()
    #create_sampling_maps(int(sys.argv[-1]))
    dets = [5, 9]			
    plt.figure(figsize=[10, 4])
    instr = 'fideos'
    for d in dets:
        sampling_maps_det(instr, d)
    #plt.xlabel(r'$s$ (pix)')
    #plt.ylabel(r'$\sigma_{RV}$ (m/s)')
    plt.ylabel(r'$s$ (pix)')
    plt.xlabel(r'$\lambda$ ($\mu m$)')
    plt.title('FIDEOS')
    plt.legend(loc='best')
    plt.savefig('/data/matala/moes/platospec/results/plots/'+instr+'_sampling_maps_dets.png')
    plt.show()
    #platospec_compare_plot()
    #number_of_ccf_lines()
