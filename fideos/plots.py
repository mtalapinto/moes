import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from optics import echelle_orders
import fideos_spectrograph
import matplotlib.cm as cm
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
import matplotlib
matplotlib.rc('font', **font)

def echellogram():
    twoddirA = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(300)), 'mm/2D/0/'])
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
    ccfAdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(300)), 'mm/ccf/'])
    ccfBdir = "".join(['/media/eduspec/TOSHIBA EXT/fideos_moes/data/f', str(int(230)), 'mm/ccf/'])
    rv = 7500
    plotfile = '/home/eduspec/Documentos/moes/fideos_moes/plots/ccf_compare_detail_'+str(int(rv))+'.png'
    ccfAdata = pd.read_csv(ccfAdir + 'ccf_'+str(int(rv))+'.tsv', sep=',')
    ccfBdata = pd.read_csv(ccfBdir + 'ccf_' + str(int(rv)) + '.tsv', sep=',')

    plt.figure(figsize=[7,4])
    plt.plot(ccfAdata['rv'], ccfAdata['ccf_norm'], 'b-', label=r'f$_{cam}$ = 300mm, samp = 2.6 pix')
    plt.plot(ccfBdata['rv'], ccfBdata['ccf_norm'], 'r-', label=r'f$_{cam}$ = 230mm, samp = 2.0 pix')
    plt.xlabel('RV (m/s)')
    plt.ylabel('Normalized CCF')
    plt.xlim(-4.5 + rv/1000, 4.5 + rv/1000)
    plt.legend()
    plt.tight_layout()
    #plt.show
    plt.savefig(plotfile)
    plt.clf()
    plt.close()


def echellogram_plot():
    rv = 0
    mask = pd.read_csv('stellar_template/g2mask_new.tsv', names=['wini', 'wend'])
    mask['wave'] = (mask['wini'] + mask['wend']) * 1e-4 / 2
    #order = echelle_orders.init_stellar_doppler_simple(rv, o)
    slit = echelle_orders.init()
    det = [0, 15., 2048, 2048]
    print(slit)
    spec = fideos_spectrograph.raytrace(slit, det)
    print(spec)
    plt.plot(spec['x'], spec['y'], 'k.')
    plt.show()


    omin = min(spec['order'])
    omax = max(spec['order'])
    order_min = min(spec['order'])
    order_max = max(spec['order'])

    cmapa = cm.nipy_spectral
    wmax, wmin = 0.7, 0.3
    fig = plt.figure(figsize=[8, 8])
    plt.title('FIDEOS')
    plt.xlabel('x (pix)')
    plt.ylabel('y (pix)')
    plt.xlim(0, 2048)
    plt.ylim(0, 2048)
    while omin <= omax:
        slit = echelle_orders.init()  # _stellar_doppler_simple(rv, instr, omin)
        specout = fideos_spectrograph.raytrace(slit, det)
        specout = specout.loc[specout['x'] >= 0]
        specout = specout.loc[specout['x'] <= det[2]]
        specout = specout.loc[specout['y'] >= 0]
        specout = specout.loc[specout['y'] <= det[3]]

        if len(specout) > 0:
            owmax = max(specout['wave'].values)
            owmin = min(specout['wave'].values)

            maskord = mask.loc[mask['wave'] < owmax]
            maskord = maskord.loc[maskord['wave'] > owmin]

            maskout = echelle_orders.init_g2mask(maskord, omin)
            specmask = fideos_spectrograph.raytrace(maskout, det)

            sx = (wmax - specout['wave'].values) / (wmax - wmin)
            plt.scatter(specout['x'].values, specout['y'].values, cmap=cmapa, c=sx, s=1, zorder=0)
            plt.plot(specmask['x'].values, specmask['y'].values, 'ko', markersize=5, zorder=10, alpha=0.6)

        omin += 1
        # print('Creating spectrum for RV = ', rv, ', order = ', order, ', samp = ', det[0])
    m = cm.ScalarMappable(cmap=cmapa)
    o = np.arange(order_min, order_max, 1)
    m.set_array(o)
    cax = fig.add_axes([0.9, .25, 0.01, 0.5])
    clb = fig.colorbar(m, orientation='vertical', cax=cax)
    clb.ax.set_ylabel('Order number', size=12)
    sampdir = '/home/marcelo/Documentos/papers/sampling/'
    plt.savefig(sampdir + 'echellogram_fideos.png')
    plt.show()



if __name__ == '__main__':

    #echellogram()
    #ccf_plot()
    echellogram_plot()
    #sampling(300)
    #sampling(230)
    #fcam = 300
    #spectrum_compare()

