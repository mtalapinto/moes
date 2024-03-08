import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import glob
import numpy as np
plt.rcParams.update({'font.family' : 'serif'})


def gaus(x, height, x0, sigma, offset):
    return height*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset


def echellogram():
    twoddirA = "".join(['/data/pix_exp/ccd_4/0/'])
    twoddirB = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(230)), 'mm/2D/0/'])
    outdir = "".join(['/home/eduspec/Documentos/moes/fideos_moes/plots/'])
    omin = 63
    omax = 104
    ms = 2
    plt.figure(figsize=[6, 6])
    while omin <= omax:
        dataA = pd.read_csv("".join([twoddirA, str(int(omin)), '_2D.tsv']), sep=',')
        dataB = pd.read_csv("".join([twoddirB, str(int(omin)), '_2D.tsv']), sep=',')

        if omin == 63:
            plt.plot(dataA['ypix'], dataA['xpix'], 'b.', alpha = 0.5, label=r'f$_{cam}$ = 300 mm, s = 2.6 pix', markersize = ms)
            plt.plot(dataB['ypix'], dataB['xpix'], 'r.', alpha = 0.5, label=r'f$_{cam}$ = 230 mm, s = 2.0 pix', markersize = ms)
        else:
            plt.plot(dataA['ypix'], dataA['xpix'], 'b.', alpha=0.5, markersize = ms)
            plt.plot(dataB['ypix'], dataB['xpix'], 'r.', alpha=0.5, markersize = ms)
        omin += 1


    plt.xlabel('x$_{CCD}$ (pix)')
    plt.xlabel('y$_{CCD}$ (pix)')
    plt.legend()
    plt.xlim(0, 2048)
    plt.ylim(0, 2048)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("".join([outdir, 'echellogram.png']))
    plt.show()


def sampling(fcam):
    basefile = "".join(['/home/eduspec/Documentos/moes/fideos_moes/hcl_spectrum/th_spec_moes_', str(int(fcam)), '.csv'])
    data = pd.read_csv(basefile, sep=',')
    print(data)
    omin = min(data['order'])
    omax = max(data['order'])
    wave, samp, R, sampeout = [], [], [], []
    fcol = 762
    slit = 0.1
    alpha = 70 * np.pi / 180
    pixsize = 15.e-3
    G = 44.41e-3  # lines/um
    d = 1/G

    while omin <= omax:
        print(omin)
        datord = data.loc[data['order'] == omin]
        waves = np.unique(datord['wave'])
        for i in range(len(waves)):
            wave.append(waves[i])
            beta = np.arcsin(omin * waves[i] / d - np.sin(alpha))
            ad = (np.sin(alpha) + np.sin(beta)) / (np.cos(beta) * waves[i])
            ld = ad * fcam
            R = (np.sin(alpha) + np.sin(beta)) * fcol / (slit * np.cos(beta))
            dlambda = waves[i] / R
            sampling = ld * dlambda / pixsize
            samp.append(sampling)

            datawave = datord.loc[datord['wave'] == waves[i]]

            xmin, xmax = int(min(datawave['x'])), int(max(datawave['x']))
            dxaux, dxaoux2, w = [], [], []

            while xmin <= xmax:
                datasamp = datawave.loc[datawave['x'] < xmin + 1]
                datasamp = datasamp.loc[datasamp['x'] > xmin]

                if len(datasamp) > 1:
                    dx = max(datasamp['y']) - min(datasamp['y'])
                    dxaux.append(dx)
                xmin += 1
            sampe = np.average(dxaux)
            sampe2 = max(dxaux)
            #print(sampe)
            sampeout.append((sampe+sampe2)/2)

        omin += 1
        sampmean = np.mean(sampeout)

        #s = (np.sin(alpha) + np.sin(beta)) * fcol /(pixsize * data['wave'].values[i] * np.cos(beta) * slit)
        #print(s)

    plt.figure(figsize=(7,4))
    plt.plot(wave, samp, 'r.', label='Theoretical sampling', alpha = 0.5, zorder = 1)
    plt.plot(wave, sampeout, 'b.', label='Moes sampling', alpha= 0.5, zorder = 0)
    plt.ylim(min(sampeout) - 0.2, max(sampeout) + 0.2)
    plt.ylabel('Spectral sampling')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.title(r'FIDEOS - f$_{cam}$ = ' + str(fcam) + ', average sampling = ' + str(np.round(np.mean(samp), 1)))
    plt.tight_layout()
    outdir = "".join(['/home/eduspec/Documentos/moes/fideos_moes/plots/'])
    plt.savefig(outdir+'sampling_f'+str(int(fcam))+'mm.png')
    #plt.show()


def spectrum_all(fcam):
    gendir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/'])
    basedir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/stellar_orders/'])
    outdir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/resolved/'])
    plotdir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/plots/'])
    moes2Ddir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/' + str(int(0)) + '/'
    pixdir = '/home/eduspec/Documentos/uai/sampling_experiment/data/f' + str(int(fcam)) + 'mm/pixelized/'
    # order =
    omin = 63
    omax = 63
    R = []
    waves = []
    while omin <= omax:
        rdata = pd.read_csv(outdir + str(omin) + '.tsv', sep=',')
        stardata = pd.read_csv(basedir + str(omin) + '.tsv', sep=',')
        moesdata = pd.read_csv(moes2Ddir + str(omin) + '_2D.tsv', sep=',')
        pixdata = pd.read_csv(pixdir + str(omin) + '.tsv', sep=',')
        moesdata['wave'] = moesdata['wave'] * 1e4
        stardata = stardata.sort_values('wave')
        rdata = rdata.sort_values('wave')

        stardata['min'] = stardata.iloc[argrelextrema(stardata.flux_norm.values, np.less_equal,
                                                      order=15)[0]]['flux_norm']
        rdata['min'] = rdata.iloc[argrelextrema(rdata.flux.values, np.less_equal, order=15)[0]]['flux']
        moesdata['min'] = moesdata.iloc[argrelextrema(moesdata.flux.values, np.less_equal, order=15)[0]]['flux']
        pixdata['min'] = pixdata.iloc[argrelextrema(pixdata.flux.values, np.less_equal, order=3)[0]]['flux']
        starmin = stardata.dropna()
        rmin = rdata.dropna()
        moesmin = moesdata.dropna()
        pixmin = pixdata.dropna()
        starmin = starmin.loc[starmin['flux_norm'] < 0.995]
        rmin = rmin.loc[rmin['flux'] < 0.995]
        moesmin = moesmin.loc[moesmin['flux'] < 0.995]
        pixmin = pixmin.loc[pixmin['flux'] < 0.995]

        plt.clf()
        plt.figure(figsize=(15, 5))
        plt.plot(rdata['wave'], rdata['flux'], 'r-', alpha=0.5)
        plt.plot(starmin['wave'], starmin['min'], 'ko', alpha=0.8)
        plt.plot(rmin['wave'], rmin['min'], 'ro', alpha=0.8)
        plt.plot(moesdata['wave'], moesdata['flux'], 'b-', alpha=0.5)
        plt.plot(moesmin['wave'], moesmin['flux'], 'bo', alpha=0.8)
        plt.plot(stardata['wave'], stardata['flux_norm'], 'k-', alpha=0.3)
        plt.plot(pixdata['wave'], pixdata['flux'], 'g-', alpha=0.5)
        plt.plot(pixmin['wave'], pixmin['flux'], 'go', alpha=0.8)
        plt.xlim(6725.7, 6726.2)
        plt.ylim(0.9781, 1.0071)
        plt.savefig('plots/compare_moes_vs_simple.png')

        # plt.plot(starmin['wave'], starmin['flux_norm'], 'ro')
        plt.show()
        plt.clf()
        plt.close()

        omin += 1


def spectrum_compare():
    fcam = 300
    fcam2 = 230
    gendir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/'])
    basedir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/stellar_orders/'])
    outdir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/resolved/'])
    plotdir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(fcam)), 'mm/plots/'])
    moes2DdirA = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/' + str(int(0)) + '/'
    moes2DdirB = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam2)) + 'mm/2D/' + str(int(0)) + '/'
    pixdir = '/home/eduspec/Documentos/uai/sampling_experiment/data/f' + str(int(fcam)) + 'mm/pixelized/'
    plotfile = '/home/eduspec/Documentos/moes/fideos_moes/plots/spec_compare_lines_detail.png'
    # order =
    omin = 63
    omax = 63
    R = []
    waves = []
    while omin <= omax:
        rdata = pd.read_csv(outdir + str(omin) + '.tsv', sep=',')
        stardata = pd.read_csv(basedir + str(omin) + '.tsv', sep=',')
        moesdataA = pd.read_csv(moes2DdirA + str(omin) + '_2D.tsv', sep=',')
        moesdataB = pd.read_csv(moes2DdirB + str(omin) + '_2D.tsv', sep=',')
        pixdata = pd.read_csv(pixdir + str(omin) + '.tsv', sep=',')
        moesdataA['wave'] = moesdataA['wave'] * 1e4
        moesdataB['wave'] = moesdataB['wave'] * 1e4
        stardata = stardata.sort_values('wave')
        rdata = rdata.sort_values('wave')

        stardata['min'] = stardata.iloc[argrelextrema(stardata.flux_norm.values, np.less_equal,
                                                      order=15)[0]]['flux_norm']
        rdata['min'] = rdata.iloc[argrelextrema(rdata.flux.values, np.less_equal, order=15)[0]]['flux']
        moesdataA['min'] = moesdataA.iloc[argrelextrema(moesdataA.flux.values, np.less_equal, order=15)[0]]['flux']
        moesdataB['min'] = moesdataB.iloc[argrelextrema(moesdataB.flux.values, np.less_equal, order=15)[0]]['flux']
        pixdata['min'] = pixdata.iloc[argrelextrema(pixdata.flux.values, np.less_equal, order=3)[0]]['flux']
        starmin = stardata.dropna()
        rmin = rdata.dropna()
        moesminA = moesdataA.dropna()
        moesminB = moesdataB.dropna()
        pixmin = pixdata.dropna()
        starmin = starmin.loc[starmin['flux_norm'] < 0.995]
        rmin = rmin.loc[rmin['flux'] < 0.995]
        moesminA = moesminA.loc[moesminA['flux'] < 0.995]
        moesminB = moesminB.loc[moesminB['flux'] < 0.995]
        pixmin = pixmin.loc[pixmin['flux'] < 0.995]

        plt.clf()
        plt.figure(figsize=(15, 5))
        #plt.plot(rdata['wave'], rdata['flux'], 'r-', alpha=0.5)
        plt.plot(starmin['wave'], starmin['min'], 'ko', alpha=0.8)
        #plt.plot(rmin['wave'], rmin['min'], 'ro', alpha=0.8)
        plt.plot(moesdataA['wave'], moesdataA['flux'], 'b-', alpha=0.5, label=r'f$_{cam}$ = 300, samp = 2.6')
        plt.plot(moesminA['wave'], moesminA['flux'], 'bo', alpha=0.8)
        plt.plot(moesdataB['wave'], moesdataB['flux'], 'r-', alpha=0.5, label=r'f$_{cam}$ = 230, samp = 2.0')
        plt.plot(moesminB['wave'], moesminB['flux'], 'ro', alpha=0.8)
        plt.plot(stardata['wave'], stardata['flux_norm'], 'k-', alpha=0.3, label='Phoenix spectrum')
        #plt.plot(pixdata['wave'], pixdata['flux'], 'g-', alpha=0.5)
        #plt.plot(pixmin['wave'], pixmin['flux'], 'go', alpha=0.8)
        # plt.savefig(plotdir+str(omin)+'.png')
        # plt.plot(starmin['wave'], starmin['flux_norm'], 'ro')
        plt.legend()
        plt.ylabel('Normalized flux')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.xlim(6769.26, 6770.04)
        plt.savefig(plotfile)
        #plt.show()
        plt.clf()
        plt.close()

        omin += 1


def ccf_plot():
    basedir = 'data/ccf/'
    ccf0dir = "".join([basedir, 'ccd_0/'])
    ccf1dir = "".join([basedir, 'ccd_1/'])
    ccf2dir = "".join([basedir, 'ccd_2/'])
    ccf3dir = "".join([basedir, 'ccd_3/'])
    ccf4dir = "".join([basedir, 'ccd_4/'])
    ccf5dir = "".join([basedir, 'ccd_5/'])
    ccf6dir = "".join([basedir, 'ccd_6/'])
    ccf7dir = "".join([basedir, 'ccd_7/'])
    ccf8dir = "".join([basedir, 'ccd_8/'])
    ccf9dir = "".join([basedir, 'ccd_9/'])
    ccf10dir = "".join([basedir, 'ccd_10/'])
    ccf11dir = "".join([basedir, 'ccd_11/'])

    rv = -10000
    plotfile = '/home/eduspec/Documentos/moes/fideos_moes/plots/ccf_compare_detail_'+str(int(rv))+'.png'
    ccftemp = pd.read_csv('ccf/ccf_0_temp_newmask.tsv', sep=',')
    ccf0data = pd.read_csv(ccf0dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf1data = pd.read_csv(ccf1dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf2data = pd.read_csv(ccf2dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf3data = pd.read_csv(ccf3dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf4data = pd.read_csv(ccf4dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf5data = pd.read_csv(ccf5dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf6data = pd.read_csv(ccf6dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf7data = pd.read_csv(ccf7dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf8data = pd.read_csv(ccf8dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf9data = pd.read_csv(ccf9dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf10data = pd.read_csv(ccf10dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')
    ccf11data = pd.read_csv(ccf11dir + 'ccf_' + str(int(rv)) + '_simple.tsv', sep=',')

    ccf0data['ccf_norm2'] = (ccf0data['ccf_norm'] - min(ccf0data['ccf_norm'].values)) / (
            max(ccf0data['ccf_norm'].values) - min(ccf0data['ccf_norm'].values))
    ccf1data['ccf_norm2'] = (ccf1data['ccf_norm'] - min(ccf1data['ccf_norm'].values)) / (
            max(ccf1data['ccf_norm'].values) - min(ccf1data['ccf_norm'].values))
    ccf2data['ccf_norm2'] = (ccf2data['ccf_norm'] - min(ccf2data['ccf_norm'].values)) / (
            max(ccf2data['ccf_norm'].values) - min(ccf2data['ccf_norm'].values))
    ccf3data['ccf_norm2'] = (ccf3data['ccf_norm'] - min(ccf3data['ccf_norm'].values)) / (
            max(ccf3data['ccf_norm'].values) - min(ccf3data['ccf_norm'].values))
    ccf4data['ccf_norm2'] = (ccf4data['ccf_norm'] - min(ccf4data['ccf_norm'].values)) / (
            max(ccf4data['ccf_norm'].values) - min(ccf4data['ccf_norm'].values))
    ccf5data['ccf_norm2'] = (ccf5data['ccf_norm'] - min(ccf5data['ccf_norm'].values)) / (
            max(ccf5data['ccf_norm'].values) - min(ccf5data['ccf_norm'].values))
    ccf6data['ccf_norm2'] = (ccf6data['ccf_norm'] - min(ccf6data['ccf_norm'].values)) / (
            max(ccf6data['ccf_norm'].values) - min(ccf6data['ccf_norm'].values))

    plt.figure(figsize=[10, 7])
    plt.plot(ccf7data['rv'], ccf7data['ccf_norm'], '-', color='black', label='s = '+str(np.round(get_det_samp(7), 2))+' pix')
    plt.plot(ccf10data['rv'], ccf10data['ccf_norm'], '-', color='gray', label='s = '+str(np.round(get_det_samp(10), 2))+' pix')
    plt.plot(ccf9data['rv'], ccf9data['ccf_norm'], '-', color='firebrick', label='s = '+str(np.round(get_det_samp(9), 2))+' pix')
    plt.plot(ccf8data['rv'], ccf8data['ccf_norm'], '-', color='red', label='s = '+str(np.round(get_det_samp(8), 2))+' pix')
    plt.plot(ccf11data['rv'], ccf11data['ccf_norm'], '-', color='orange', label='s = '+str(np.round(get_det_samp(11), 2))+' pix')
    plt.plot(ccf0data['rv'], ccf0data['ccf_norm'], '-', color='gold',  label='s = '+str(np.round(get_det_samp(0), 2))+' pix')
    plt.plot(ccf1data['rv'], ccf1data['ccf_norm'], '-', color='greenyellow', label='s = '+str(np.round(get_det_samp(1), 2))+' pix')
    plt.plot(ccf2data['rv'], ccf2data['ccf_norm'], '-', color='darkgreen', label='s = '+str(np.round(get_det_samp(2), 2))+' pix')
    plt.plot(ccf3data['rv'], ccf3data['ccf_norm'], '-', color = 'darkcyan',label='s = '+str(np.round(get_det_samp(3), 2))+' pix')
    plt.plot(ccf4data['rv'], ccf4data['ccf_norm'], '-', color='deepskyblue',label='s = '+str(np.round(get_det_samp(4), 2))+' pix')
    plt.plot(ccf5data['rv'], ccf5data['ccf_norm'], '-', color='blue',label='s = '+str(np.round(get_det_samp(5), 2))+' pix')
    plt.plot(ccf6data['rv'], ccf6data['ccf_norm'], '-', color='purple', label='s = '+str(np.round(get_det_samp(6), 2))+' pix')


    plt.xlabel('RV (km/s)')
    plt.ylabel('CCF')
    plt.xlim(-15 + rv/1000, 15 + rv/1000)
    plt.ylim(0.8, 4.3)
    plt.legend(loc=3, ncol=2)
    plt.tight_layout()
    #plt.show()
    plt.savefig('ccf_compare.png')
    plt.clf()
    plt.close()


def order_plot(fcam):
    basedir = "".join(['/home/eduspec/Documentos/moes/platospec/'])
    moes2Ddir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/2D/' + str(int(0)) + '/'
    fideosdir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(300)) + 'mm/2D/' + str(int(0)) + '/'

    #stardata = pd.read_csv(basedir + 'stellar_spectrum_norm.dat', sep=',')
    stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    stardata_0 = pd.read_csv(stardir+'P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec',
                             delim_whitespace=True,
                             names=['wave', 'flux', 'flux_norm'])
    stardata_1 = pd.read_csv(
        '/home/eduspec/Documentos/moes/M_s5750g3.0z0.00t2.0_a-0.40c0.00n0.00o-0.40r0.00s0.00_VIS.spec.flat/M_s5750g3.0z0.00t2.0_a-0.40c0.00n0.00o-0.40r0.00s0.00_VIS.spec',
        delim_whitespace=True,
        names=['wave', 'flux', 'flux_norm'])

    pixeldir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(300)), 'mm/pixelized/'])
    resoldir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/data/f', str(int(300)), 'mm/resolved/'])

    plpixeldir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/platospec/f', str(int(360)), 'mm/pixelized/'])
    plresoldir = "".join(['/home/eduspec/Documentos/uai/sampling_experiment/platospec/f', str(int(360)), 'mm/resolved/'])

    moes2Ddir2 = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(240)) + 'mm/' + str(int(0)) + '/'

    order = 100
    ordata = pd.read_csv(moes2Ddir + str(int(order))+'_2D.tsv', sep=',')
    ordatab = pd.read_csv(moes2Ddir + str(int(order)) + '_2D_v2.tsv', sep=',')
    ordatac = pd.read_csv(moes2Ddir + str(int(order)) + '_2D_v2_red.tsv', sep=',')
    firorder = 90
    fidata = pd.read_csv(fideosdir + str(int(firorder)) + '_2D_v2_red.tsv', sep=',')
    fidata_0 = pd.read_csv(fideosdir + str(int(firorder)) + '_2D.tsv', sep=',')

    fideos_pixelized = pd.read_csv(pixeldir+str(int(firorder))+'.tsv', sep=',')
    fideos_resolved = pd.read_csv(resoldir + str(int(firorder)) + '.tsv', sep=',')

    platospec_pixelized = pd.read_csv(plpixeldir + str(int(order)) + '.tsv', sep=',')
    platospec_resolved = pd.read_csv(plresoldir + str(int(order)) + '.tsv', sep=',')

    platospec_f240 = pd.read_csv(moes2Ddir2 + str(int(order)) + '.tsv', sep=',')

    ordata['wave'] = ordata['wave'] * 1e4
    ordatab['wave'] = ordatab['wave'] * 1e4
    ordatac['wave'] = ordatac['wave'] * 1e4
    fidata['wave'] = fidata['wave'] * 1e4
    fidata_0['wave'] = fidata_0['wave'] * 1e4
    wmin = min(ordata['wave'])
    wmax = max(ordata['wave'])

    stardata = stardata.loc[stardata['wave'] < wmax]
    stardata = stardata.loc[stardata['wave'] > wmin]
    stardata = stardata.sort_values('wave')

    stardata['min'] = stardata.iloc[argrelextrema(stardata.flux_norm.values, np.less_equal,
                                                  order=5)[0]]['flux_norm']
    starmin = stardata.dropna()
    starmin = starmin.loc[starmin['flux_norm'] < 1.]
    ordata['min'] = ordata.iloc[argrelextrema(ordata.flux.values, np.less_equal,
                                                  order=5)[0]]['flux']
    ordamin = ordata.dropna()
    ordamin = ordamin.loc[ordamin['flux'] < 1.]

    ordatac['min'] = ordatac.iloc[argrelextrema(ordatac.flux.values, np.less_equal,
                                                order=5)[0]]['flux']
    ordcmin = ordatac.dropna()
    ordcmin = ordcmin.loc[ordcmin['flux'] < 1.]

    #plt.plot(starmin['wave'], starmin['flux_norm'],'ko')
    #plt.plot(ordamin['wave'], ordamin['flux'], 'ro')
    #plt.plot(ordcmin['wave'], ordcmin['flux'], 'bo')
    #plt.plot(stardata['wave'], stardata['flux_norm'], 'k-', alpha=0.5)
    stardata['wavediff'] = stardata['wave'].diff()
    stardata = stardata.dropna()
    stardata_0['wavediff'] = stardata_0['wave'].diff()
    stardata_0 = stardata_0.dropna()
    stardata_1['wavediff'] = stardata_1['wave'].diff()
    stardata_1 = stardata_1.dropna()
    #print(stardata['wave']/stardata['wavediff'])
    print(stardata_0['wave'] / stardata_0['wavediff'])
    #print(stardata_1['wave'] / stardata_1['wavediff'])
    #plt.plot(ordata['wave'], ordata['flux'], 'r-', alpha=0.8)
    #plt.plot(ordatac['wave'], ordatac['flux'], 'r--', alpha=0.4)
    #plt.plot(fidata['wave'], fidata['flux'], 'b--', alpha=0.4)
    #plt.plot(fidata_0['wave'], fidata_0['flux'], 'b--', alpha=0.4)
    plt.plot(stardata_0['wave'], stardata_0['flux_norm'], 'k-', alpha=0.4)
    #plt.plot(fideos_resolved['wave'], fideos_resolved['flux'], 'g-', alpha=0.8)
    #plt.plot(fideos_pixelized['wave'], fideos_pixelized['flux'], 'g--', alpha=0.4)
    #plt.plot(platospec_pixelized['wave'], platospec_pixelized['flux'], 'm--', alpha=0.8)
    plt.plot(platospec_f240['wave'], platospec_f240['flux'], 'm--', alpha=0.8)
    #plt.plot(stardata_1['wave'], stardata_1['flux_norm'], 'b-', alpha=0.5)
    #plt.xlim(min(ordata['wave']), max(ordata['wave']))
    plt.show()
    plt.close()

    Rori, wavesori = [], []
    for i in range(len(starmin)):
        wcen = starmin['wave'].values[i]
        dlambda = 0.12
        linedata = stardata.loc[stardata['wave'] < wcen + dlambda]
        linedata = linedata.loc[linedata['wave'] > wcen - dlambda]
        plt.clf()
        plt.plot(linedata['wave'], linedata['flux_norm'])
        #plt.show()
        plt.close()
        linedata['min'] = linedata.iloc[argrelextrema(linedata.flux_norm.values, np.greater_equal, order=5)[0]]['flux_norm']
        linemins = linedata.dropna()

        if len(linemins) > 1:
            linemins = linemins.sort_values('wave')
            linedata = linedata.loc[linedata['wave'] < max(linemins['wave'])]
            linedata = linedata.loc[linedata['wave'] > min(linemins['wave'])]
            linedata['flux_norm2'] = (linedata['flux_norm'] - min(linedata['flux_norm'])) / (
                    max(linedata['flux_norm']) - min(linedata['flux_norm']))

            x = linedata['wave']
            y = linedata['flux_norm2']

            # g_init = models.Gaussian1D(amplitude=-1., mean=wcen, stddev=dlambda)
            # fit_g = fitting.LevMarLSQFitter()
            # g = fit_g(g_init, x, y)
            height = -1
            mean = wcen
            sigma = dlambda
            offset = 1.
            try:
                popt, pcov = curve_fit(gaus, x, y, p0=[height, mean, sigma, offset])
                # perr = np.sqrt(np.diag(pcov))
                fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                res = wcen / fwhm
                #print(np.abs(res))
                Rori.append(np.abs(res))
                wavesori.append(wcen)

            except RuntimeError:
                print('Error in curve fitting...')
    
    Ra, wavesa = [], []
    for i in range(len(ordamin)):
        wcen = ordamin['wave'].values[i]
        dlambda = 0.2
        linedata = ordata.loc[ordata['wave'] < wcen + dlambda]
        linedata = linedata.loc[linedata['wave'] > wcen - dlambda]
        plt.clf()
        plt.plot(linedata['wave'], linedata['flux'])
        #plt.show()
        plt.close()
        linedata['min'] = linedata.iloc[argrelextrema(linedata.flux.values, np.greater_equal, order=5)[0]][
            'flux']
        linemins = linedata.dropna()

        if len(linemins) > 1:
            linemins = linemins.sort_values('wave')
            linedata = linedata.loc[linedata['wave'] < max(linemins['wave'])]
            linedata = linedata.loc[linedata['wave'] > min(linemins['wave'])]
            linedata['flux_norm'] = (linedata['flux'] - min(linedata['flux'])) / (
                    max(linedata['flux']) - min(linedata['flux']))

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
                # perr = np.sqrt(np.diag(pcov))
                fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                res = wcen / fwhm
                print(np.abs(res))
                Ra.append(np.abs(res))
                wavesa.append(wcen)

            except RuntimeError:
                print('Error in curve fitting...')


    Rc, wavesc = [], []
    for i in range(len(ordcmin)):
        wcen = ordcmin['wave'].values[i]
        dlambda = 0.1
        linedata = ordatac.loc[ordatac['wave'] < wcen + dlambda]
        linedata = linedata.loc[linedata['wave'] > wcen - dlambda]
        plt.clf()
        plt.plot(linedata['wave'], linedata['flux'])
        #plt.show()
        plt.close()
        linedata['min'] = linedata.iloc[argrelextrema(linedata.flux.values, np.greater_equal, order=5)[0]][
            'flux']
        linemins = linedata.dropna()

        if len(linemins) > 1:
            linemins = linemins.sort_values('wave')
            linedata = linedata.loc[linedata['wave'] < max(linemins['wave'])]
            linedata = linedata.loc[linedata['wave'] > min(linemins['wave'])]
            linedata['flux_norm'] = (linedata['flux'] - min(linedata['flux'])) / (
                    max(linedata['flux']) - min(linedata['flux']))

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
                # perr = np.sqrt(np.diag(pcov))
                fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                res = wcen / fwhm
                print(np.abs(res))
                Rc.append(np.abs(res))
                wavesc.append(wcen)

            except RuntimeError:
                print('Error in curve fitting...')


    plt.plot(wavesc, Rc, 'k.')
    plt.plot(wavesori, Rori, 'r.')
    plt.plot(wavesa, Ra, 'b.')
    #plt.ylim(50000, 100000)
    plt.show()


def moes_plot(fcam):
    #moesdir = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/moes/' + str(int(0)) + '/'
    moesdir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/moes/' + str(int(0)) + '/'
    order = 114

    ordata = pd.read_csv(moesdir + str(int(order)) + '.tsv', sep=',')
    waves = np.unique(ordata['wave'])
    line = ordata.loc[ordata['wave'] == waves[10]]
    #plt.plot(ordata['x'].astype(float), ordata['y'].astype(float), 'k.')
    plt.plot(line['x'].astype(float), line['y'].astype(float), 'k.')
    plt.show()


def slit_plot(fcam):
    #moesdir = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/moes/' + str(int(0)) + '/'
    moesdir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(0)) + '/'
    order = 100
    ordata = pd.read_csv(moesdir + str(int(order)) + '_slit.tsv', sep=',')
    waves = np.unique(ordata['wave'])
    line = ordata.loc[ordata['wave'] == waves[-1]]
    #plt.plot(ordata['x'].astype(float), ordata['y'].astype(float), 'k.')
    plt.plot(line['x'].astype(float), line['y'].astype(float), 'k.')
    plt.show()


def new_order_plot():
    fideos_dir = '/home/eduspec/Documentos/moes/fideos_moes/data/f300mm/2D/0/'
    orderfid = 64
    fideosdata = pd.read_csv(fideos_dir + str(int(orderfid))+'_2D.tsv', sep=',')
    stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    stardata = pd.read_csv(stardir + 'P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec',
                             delim_whitespace=True,
                             names=['wave', 'flux', 'flux_norm'])
    moes2Ddir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(240)) + 'mm/' + str(int(0)) + '/'
    order = 70
    platospec_f240 = pd.read_csv(moes2Ddir + str(int(order)) + '_2D.tsv', sep=',')
    platospec_f240 = platospec_f240.sort_values('wave')
    stardata = stardata.sort_values('wave')
    plt.plot(stardata['wave'], stardata['flux_norm'], 'k-', alpha=0.5)
    plt.plot(platospec_f240['wave']*1e4, platospec_f240['flux'], 'r-', alpha=0.5)
    plt.plot(fideosdata['wave'] * 1e4, fideosdata['flux'], 'b-', alpha=0.5)
    plt.show()


def ccd_exp_spectrum_plot():
    import ccf
    stardata = pd.read_csv('stellar_template/stellar_template.tsv', sep=',')
    stardata = stardata.sort_values('wave')
    maskdata = pd.read_csv('stellar_template/g2mask_new.tsv', sep=',', names=['wini', 'wend'])
    #print(maskdata)
    order = 86
    det0data = pd.read_csv('data/pix_exp/ccd_0/0/'+str(order)+'_2D_moes.tsv', sep=',')
    det4data = pd.read_csv('data/pix_exp/ccd_4/0/' + str(order) + '_2D_moes.tsv', sep=',')
    det6data = pd.read_csv('data/pix_exp/ccd_6/0/' + str(order) + '_2D_moes.tsv', sep=',')
    wmin, wmax = min(det0data['wave'])*1e4, max(det0data['wave'])*1e4
    #print(len(maskdata))
    maskdata = maskdata.loc[maskdata['wini'] > wmin]
    maskdata = maskdata.loc[maskdata['wend'] < wmax]
    print(len(maskdata))
    for k in range(len(maskdata)):
        xmin = maskdata['wini'].values[k]
        xmax = maskdata['wend'].values[k]
        x = [xmin, xmax]
        plt.fill_between(x, -0.5, 1.4, color='gray', alpha=0.4)

    plt.plot(stardata['wave'], stardata['flux'], 'k-', alpha=0.8, label='Stellar template')
    plt.plot(det0data['wave'] * 1e4, det0data['flux'], 'r-o', alpha=0.5,
             label='s = ' + str(np.round(get_det_samp(0), 2)) + ' pix')
    plt.plot(det4data['wave'] * 1e4, det4data['flux'], 'g-o', alpha=0.5,
             label='s = ' + str(np.round(get_det_samp(4), 2)) + ' pix')

    plt.plot(det6data['wave']*1e4, det6data['flux'], 'b-o', alpha=0.5, label='s = '+str(np.round(get_det_samp(6), 2))+' pix')
    plt.xlim(5421.08, 5421.27)
    plt.ylim(0.6, 1)
    plt.xlabel(r'Wavelength ($\AA{}$)')
    plt.ylabel('Normalized flux')
    plt.legend(loc='best')
    plt.savefig('line_example_mask.png')
    #plt.show()


def compare_ccf():
    rvs = np.arange(-10000, -1, 250)

    datadir = 'data/pix_exp/'
    for rv in rvs:
        ccd0data = pd.read_csv(datadir+'ccd_0/ccf/ccf_'+str(rv)+'_obs.tsv',sep=',')
        ccd1data = pd.read_csv(datadir + 'ccd_1/ccf/ccf_' + str(rv) + '_obs.tsv', sep=',')
        plt.plot(ccd0data['rv'] - rv*1e-3, ccd0data['ccf_norm'], 'r-', alpha=0.3)
        plt.plot(ccd1data['rv'] - rv*1e-3, ccd1data['ccf_norm'], 'b-', alpha=0.43)
    plt.show()


def echellogram_plot():
    import echelle_orders
    import platospec_moes
    import matplotlib.cm as cm
    rv = 0
    instr = 'platospec'
    omin = 68
    omax = 122
    det = [2.02, 13.5, 2048, 2048]
    cmap = cm.get_cmap('Spectral')
    wmax, wmin = 0.7, 0.3
    fig = plt.figure(figsize=[8,8])
    while omin <= omax:
        slitout = echelle_orders.init()  # _stellar_doppler_simple(rv, instr, omin)
        specout = platospec_moes.tracing_full_det(slitout, det)
        specout = specout.loc[specout['x'] >= 0]
        specout = specout.loc[specout['x'] <= det[2]]
        specout = specout.loc[specout['y'] >= 0]
        specout = specout.loc[specout['y'] <= det[3]]
        sx = (wmax - specout['wave'].values)/(wmax - wmin)
        plt.scatter(specout['x'].values, specout['y'].values, cmap=cm.Spectral, c=sx, s=1)
        omin += 1
    #print('Creating spectrum for RV = ', rv, ', order = ', order, ', samp = ', det[0])
    m = cm.ScalarMappable(cmap=cm.Spectral)
    o = np.arange(68, 122, 1)
    m.set_array(o)
    cax = fig.add_axes([0.9, .25, 0.01, 0.5])
    clb = fig.colorbar(m, orientation='vertical', cax=cax)
    clb.ax.set_ylabel('Order number', size=12)

    plt.savefig('echellogram_platospec.png')
    plt.show()
    # print(slitout)

    # Doing Moe's ray tracing
    print('Ray tracing with moes... ', )

    print('done')


def template_plot():
    spec = pd.read_csv()
    data = pd.read_csv('stellar_template/stellar_template_v2.tsv', sep=' ')
    print(data)
    plt.figure(figsize=[8,3])
    plt.plot(data['WAVE'].values, data['FLUX'], 'k-')
    plt.xlim(5520, 5560)
    plt.xlabel(r'Wavelength ($\AA$)')
    plt.ylabel(r'Normalized flux')
    plt.tight_layout()
    plt.savefig('template_spec.png')
    plt.clf()
    # plt.show()


def get_det_samp(i, sn):
    if sn == 0:
        datadir = '/data/matala/luthien/platospec/data/pix_exp/ns'+str(int(sn))+'/ccd_'+str(i)+'/0/'
        orand = 90
        data = pd.read_csv(datadir + str(int(orand)) + '_2D_moes.tsv', sep=',')

    elif sn == 1 or sn == 5 or sn == 10:
        datadir = '/data/matala/luthien/platospec/data/pix_exp/ns'+str(int(sn))+'/ccd_' + str(i) + '/0/'
        orand = 90
        data = pd.read_csv(datadir + str(int(orand)) + '_2D_moes.tsv', sep=',')

    npix = len(data)
    ccdx = 2048 * 13.5  # um
    pixsize = ccdx / npix

    fcam = 240
    fcol = 876
    slit = 100
    samp = slit*fcam/fcol/pixsize
    return samp


def single_ccf_plot(det, rv, sn):
    basedir = '/melian/moes/platospec/data/ccf/'
    ccf = pd.read_csv(basedir + 'sn'+str(int(sn))+ '/ccd_' + str(int(det)) +'/ccf_'+str(int(rv))+'_simple.tsv', sep=',')
    ccf2 = pd.read_csv(basedir + 'ccd_' + str(int(det)) + '/ccf_' + str(int(rv)) + '_simple.tsv',
                      sep=',')
    plt.plot(ccf['rv'], ccf['ccf'], 'k-', alpha=0.5)
    plt.plot(ccf2['rv'], ccf2['ccf'], 'r-', alpha=0.8)
    plt.xlabel('RV [km/s]')
    plt.ylabel('CCF')
    plt.tight_layout()
    plt.show()


def spectrum_plot():
    rv = str(int(-10000))
    det = str(int(0))
    basedir = '/luthien/platospec/data/pix_exp/'
    simpledir = basedir + 'ns0/ccd_' + det + '/' + rv + '/'
    fulldir = basedir + 'ns0_full/ccd_' + det + '/' + rv + '/'
    order = str(80)
    simpdata = pd.read_csv(simpledir + order + '_2D_moes.tsv', sep=',')
    fulldata = pd.read_csv(fulldir + order + '_2D_moes.tsv', sep=',')

    plt.plot(simpdata['wave'], simpdata['flux'], 'r.-', alpha=0.5)
    plt.plot(fulldata['wave'], fulldata['flux'], 'b.-', alpha=0.5)
    plt.show()
    print(simpdata)
    print(fulldata)


def ccf_test_plot():

    basedir = '/luthien/platospec/data/ccf/ns0_full/ccd_0/'
    data = pd.read_csv(basedir + 'ccf_-10000_full.tsv', sep = ',')

    plt.plot(data['rv'], data['ccf'], 'k-')
    plt.show()


def sigma_plots(instr):
    if instr == 'carmenes':
        basedir = '/home/marcelo/Documentos/moes/' + str(instr) + '/vis/'
    else:
        basedir = '/home/marcelo/Documentos/moes/' + str(instr) + '/'

    results = basedir+'results/'

    nss = [0, 1, 5, 10]
    ns0, ns1, ns5, ns10 = [], [], [], []
    colors = ['orangered', 'gold', 'limegreen', 'cornflowerblue']
    i = 0
    plt.figure(figsize=[6, 4])
    for ns in nss:
        drvfiles = glob.glob(results+'ns'+str(ns)+'/*')
        #print(drvfiles)
        stdaux, sampaux = [], []
        dataplot = pd.DataFrame()
        for drvfile in drvfiles:
            data = pd.read_csv(drvfile, sep=',')
            std = np.std(data['drv'].values)
            samp = data['samp'].values[0]
            if drvfile == '/home/marcelo/Documentos/moes/feros/results/ns1/drv_ccd_0_simple.tsv':
                stdaux.append(1.421)
                sampaux.append(samp)
            else:
                stdaux.append(std)
                sampaux.append(samp)

        dataplot['std'] = stdaux
        dataplot['samp'] = sampaux
        dataplot = dataplot.sort_values(by='samp')
        plt.plot(dataplot['samp'], dataplot['std'], 'o-', color=colors[i], label='S/N = '+str(ns)+'%')
        i += 1

    outdir = '/home/marcelo/Documentos/moes/sampling_plots/'
    plt.xlabel('Sampling [pix]')
    plt.ylabel(r'$\sigma_{RV}$ [m/s]')
    plt.title(instr.capitalize())
    plt.legend(loc='best')
    plt.savefig(outdir+instr+'_sn.png')
    plt.show()

        #data = pd.read_csv()
        #ns0.append()


def instru_plots(ns):
    instruments = ['platospec', 'fideos', 'carmenes', 'feros']
    #basedir = '/home/marcelo/Documentos/moes/'
    basedir = '/data/matala/moes/'
    i = 0
    colors = ['orangered', 'gold', 'limegreen', 'cornflowerblue']
    for instr in instruments:
        instrudir = basedir + instr + '/results/ns'+str(ns)+'/'
        drvfiles = glob.glob(instrudir + '*')
        # print(drvfiles)
        stdaux, sampaux = [], []
        dataplot = pd.DataFrame()
        for drvfile in drvfiles:
            data = pd.read_csv(drvfile, sep=',')
            std = np.std(data['drv'].values)
            samp = data['samp'].values[0]
            if drvfile == basedir + 'feros/results/ns1/drv_ccd_0_simple.tsv':
                stdaux.append(1.421)
                sampaux.append(samp)
            else:
                stdaux.append(std)
                sampaux.append(samp)
        dataplot['std'] = stdaux
        dataplot['samp'] = sampaux
        dataplot = dataplot.sort_values(by='samp')
        plt.plot(dataplot['samp'], dataplot['std'], 'o-', color=colors[i], label=instr.capitalize())
        i += 1

    outdir = basedir + 'sampling_plots/'
    plt.xlabel('Sampling [pix]')
    plt.ylabel(r'$\sigma_{RV}$ [m/s]')
    plt.title('S/N = '+str(ns)+'%')
    plt.legend(loc='best')
    plt.savefig(outdir + 'ston_'+str(ns)+'.png')
    plt.show()



def histo_plots(instr):
    
    basedir = "/data/matala/moes/" + str(instr) + "/results/"
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
        n = 10
        imccd = slit * fcam / fcol
        rv = 0
        dets = [1, 9]
        #idet = pd.read_csv('/data/matala/luthien/'+str(instr)+'/data/pix_exp/ns0/ccd_'+str(int(i))+'/0/'+str(int(omin))+'_2D_moes.tsv', sep=',')

    elif instr == 'carmenes':
        pixdir = datadir + 'pix_exp/'
        fcam = 455.
        fcol = 1590.
        slit = 146.
        omin = 87
        omax = 171
        x_um = 4096. * 15.0
        y_um = 4096. * 15.0
        pixarray = np.arange(4.5, 24.1, 1.5)
        #n = len(pixarray)
        n = 13
        title = 'CARMENES'
        imccd = slit * fcam / fcol
        rv = 0
        dets = [1, 12]
        #idet = pd.read_csv('/data/matala/luthien/'+str(instr)+'/data/pix_exp/ns0/ccd_'+str(int(i))+'/0/'+str(int(omin))+'_2D_moes.tsv', sep=',')
    elif instr == 'feros':
        pixdir = datadir + 'data/pix_exp/'
        omin = 32
        omax = 60
        fcam = 410.
        fcol = 1501.
        slit = 120
        x_um = 4096 * 15  # um
        y_um = 2048 * 15
        pixarray = np.arange(4.5, 19.6, 1.5)
        #n = len(pixarray)
        n = 10
        title = 'FEROS'
        imccd = slit * fcam / fcol
        rv = 0
        dets = [1, 8]
        #idet = pd.read_csv('/data/matala/luthien/'+str(instr)+'/data/pix_exp/ns0/ccd_'+str(int(i))+'/0/'+str(int(omin))+'_2D_moes.tsv', sep=',')
    
    elif instr == 'fideos':
        #n = 9
        title = 'FIDEOS'
        x_um = 2048 * 13.5
        y_um = 2048 * 13.5
        omin = 64
        omax = 106
        fcam = 300.
        fcol = 762.
        slit = 100
        pixarray = np.arange(4.5, 22.6, 1.5)
        n = len(pixarray)
        n = 10
        pixdir = datadir + 'data/'
        imccd = slit * fcam / fcol
        dets = [1, 8]
        rv = 0
    
    dets = [5]
    resdir = basedir + 'ns0/'
    res2dir = basedir + 'ns10/'
    bins = np.arange(-10, 10, 1.)
    colors = ['gold', 'blue', 'gray','red']
    i = 0
    plt.figure(figsize=[7,3])
    for det in dets:
        resdata = pd.read_csv(resdir + 'drv_ccd_' + str(int(det)) + '_simple.tsv', sep=',')
        resdata2 = pd.read_csv(res2dir + 'drv_ccd_' + str(int(det)) + '_simple.tsv', sep=',')
        samp = np.mean(resdata['samp'].values)
        plt.hist(resdata['drv'] - np.mean(resdata['drv']), bins=bins, color=colors[0], alpha=0.5, ec='black', label='s = 2.2 pix, no noise')
        plt.hist(resdata2['drv'] - np.mean(resdata2['drv']), bins=bins, color=colors[1], alpha=0.5, ec='black', label='s = 2.2 pix, 10% noise')
        i += 1
    
    plt.ylabel('Number of observations')
    plt.xlabel(r'$\Delta$RV (m/s)')
    plt.legend(loc='best')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('results/plots/'+str(instr)+'_histo_compare_noise.png')
    plt.show()



def plots_tilt():
    rv = -1500
    basedir = '/data/matala/luthien/platospec/data/ccf/ns0_full/ccd_5/'
    basedir2 = '/data/matala/luthien/platospec/data/ccf/ns0/ccd_5/'
    ccf1 = pd.read_csv(basedir + 'ccf_' + str(rv) + '_full.tsv', sep=',')
    ccf2 = pd.read_csv(basedir2 + 'ccf_' + str(rv) + '_simple.tsv', sep=',')
    
    plt.figure(figsize = [8, 4])
    plt.plot(ccf1['rv'], ccf1['ccf'], 'r-', label = 'with tilt ')
    plt.plot(ccf2['rv'], ccf2['ccf'], 'b-', label = 'No tilt')
    plt.xlabel('RV (km/s)')
    plt.ylabel('CCF')
    plt.legend(loc='best')
    plt.savefig('results/plots/ccf_tilt_compare.png')
    plt.show()

    basedir = '/data/matala/luthien/platospec/data/pix_exp/ns0_full/ccd_5/'
    basedir2 = '/data/matala/luthien/platospec/data/pix_exp/ns0/ccd_5/'
    plt.figure(figsize = [8, 4])
    stardata = pd.read_csv('stellar_template/stellar_template.tsv', sep=',')
    stardata = stardata.sort_values('wave')
    maskdata = pd.read_csv('stellar_template/g2mask_new.tsv', sep=',', names=['wini', 'wend'])
    #print(maskdata)
    order = 86
    det0data = pd.read_csv(basedir + '0/'+str(order)+'_2D_moes.tsv', sep=',')
    det4data = pd.read_csv(basedir2 + '0/' + str(order) + '_2D_moes.tsv', sep=',')
    #det6data = pd.read_csv('data/pix_exp/ccd_6/0/' + str(order) + '_2D_moes.tsv', sep=',')
    wmin, wmax = min(det0data['wave'])*1e4, max(det0data['wave'])*1e4
    #print(len(maskdata))
    maskdata = maskdata.loc[maskdata['wini'] > wmin]
    maskdata = maskdata.loc[maskdata['wend'] < wmax]
    print(len(maskdata))
    for k in range(len(maskdata)):
        xmin = maskdata['wini'].values[k]
        xmax = maskdata['wend'].values[k]
        x = [xmin, xmax]
        plt.fill_between(x, -0.5, 1.4, color='gray', alpha=0.4)

    plt.plot(stardata['wave'], stardata['flux'], 'k-', alpha=0.8, label='Stellar template')
    plt.plot(det0data['wave'] * 1e4, det0data['flux'], 'r-o', alpha=0.5,
             label='with tilt')
    plt.plot(det4data['wave'] * 1e4, det4data['flux'], 'b-o', alpha=0.5,
             label='No tilt')

    #plt.plot(det6data['wave']*1e4, det6data['flux'], 'b-o', alpha=0.5, label='s = '+str(np.round(get_det_samp(6), 2))+' pix')
    plt.xlim(5421.04, 5421.31)
    plt.ylim(0.6, 1)
    plt.xlabel(r'Wavelength ($\AA{}$)')
    plt.ylabel('Normalized flux')
    plt.legend(loc='best')
    plt.savefig('results/plots/line_example_mask_tilt.png')
    plt.show()


def nonoiseplot():
    basedir = '/data/matala/moes/'
    platospec = basedir + 'platospec/results/ns0/'
    npl = 10
    carmenes = basedir + 'carmenes/results/ns0/'
    ncr = 12
    feros = basedir + 'feros/results/ns0/'
    nfr = 10
    fideos = basedir + 'fideos/results/ns0/'
    nfd = 11
    drv, samp = [], []
    for i in range(ncr):
        drvfile = pd.read_csv(carmenes + 'drv_ccd_'+str(i)+'_simple.tsv', sep=',')
        drv.append(np.std(drvfile['drv']))
        samp.append(np.mean(drvfile['samp']))
    fig = plt.figure(figsize=[6, 4])
    plt.plot(samp, drv, 'o-', c='cornflowerblue', label=r'CARMENES, R$\sim$85\,000')    
    #plt.show()
    #plt.close()
    
    drv, samp = [], []
    for i in range(npl):
        drvfile = pd.read_csv(platospec + 'drv_ccd_'+str(i)+'_simple.tsv', sep=',')
        drv.append(np.std(drvfile['drv']))
        samp.append(np.mean(drvfile['samp']))
    plt.plot(samp, drv, 'o-', c='gold', label=r'PLATOSpec, R$\sim$70\,000')    
    
    drv, samp = [], []
    for i in range(nfr):
        drvfile = pd.read_csv(feros + 'drv_ccd_'+str(i)+'_simple.tsv', sep=',')
        drv.append(np.std(drvfile['drv']))
        samp.append(np.mean(drvfile['samp']))
    plt.plot(samp, drv, 'o-', c='gray', label=r'FEROS, R$\sim$48\,000')    
    

    drv, samp = [], []
    for i in range(nfd):
        drvfile = pd.read_csv(fideos + 'drv_ccd_'+str(i)+'_simple.tsv', sep=',')
        drv.append(np.std(drvfile['drv']))
        samp.append(np.mean(drvfile['samp']))
    aux = pd.DataFrame()
    aux['samp'] = samp
    aux['drv'] = drv
    aux = aux.sort_values(by='samp') 
    plt.plot(aux['samp'], aux['drv'], 'o-', c='red', label=r'FIDEOS, R$\sim$42\,000')    
    
    plt.show()
    plt.close()
    

    #drv, samp = [], []
    #for i in range(ncr):
    #    drvfile = pd.read_csv(carmenes + 'drv_ccd_'+str(i)+'_simple.tsv', sep=',')
    #    drv.append(np.std(drvfile['drv']))
    #    samp.append(np.mean(drvfile['samp']))
        
    
if __name__ == '__main__':


    #echellogram()
    #ccf_plot()
    #moes_plot(360)
    #new_order_plot()
    #ccd_exp_spectrum_plot()
    #ccf_plot()
    #template_plot()
    #ccf_plot()
    #ccf_test_plot()
    #instrument = 'carmenes'
    #sigma_plots(instrument)
    #instru_plots(10)
    carlos = 'juan'
    #nonoiseplot()
    #histo_plots('platospec')
    #histo_plots('fideos')
    #histo_plots('carmenes')
    #histo_plots('feros')
    
    #plots_tilt()
    #print(get_det_samp(4))
    #single_ccf_plot(0, -10000, 1)
    #spectrum_plot()
    #echellogram_plot()
    #compare_ccf()
    #order_plot(240)
    # moes_plot(360)
    #spectrum_all(300)
    #sampling(300)
    #sampling(230)
    #fcam = 300
    #spectrum_compare()

