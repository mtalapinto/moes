import numpy as np
import pandas as pd
import echelle_orders
import fideos_spectrograph
import glob
import os
import ccf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaus(x, height, x0, sigma):  #, offset):
    return height*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) #+ offset


def create_rv_slit_files(fcam):
    # RV range to shift the spectrum
    rvrange = np.arange(-10000, 10000, 250)
    print(len(rvrange))
    # create echelle orders w/slit coordinates
    outpath = '/home/eduspec/Documentos/moes/fideos_moes/data/f'+str(int(fcam))+'mm/'
    for rv in rvrange:
        echelle_orders.init_stellar_doppler(rv, fcam)
    print('Slit file written... rv = ', rv, 'fcam = ', fcam, '\n')


def create_moes_spectrum(fcam):
    # basedir = '/home/eduspec/Documentos/moes/fideos_moes/stellar_spectrum/f230mm/rv_specs/'
    basedir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f'+str(int(fcam))+'mm/rv_specs/'

    files = glob.glob(basedir+'*.dat')

    for file in files:
        print('Creating MOES spectrum...')
        rv = file[len(basedir) + 21:-4]
        print(rv)
        outpath = basedir + str(int(rv)) + '/'

        if not os.path.exists(outpath):
            os.makedirs(outpath)
            data = pd.read_csv(file, sep=',')
            # print(data)
            omin = min(data['order'])
            omax = max(data['order'])

            while omin <= omax:
                print(fcam, rv, omin)
                datord = data.loc[data['order'] == omin]
                specmoes = fideos_spectrograph.tracing_full_fcam(datord, fcam)
                specmoes.to_csv(outpath + str(int(omin)) + '.csv', index=False)
                omin += 1

        else:
            print('Files already created for rv = ', int(rv))


def create_2D_spectra(fcam):
    #basedir = '/home/eduspec/Documentos/moes/fideos_moes/stellar_spectrum/f230mm/rv_specs/'
    basedir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/rv_specs/'
    dirs = glob.glob(basedir+'*/')
    for dir in dirs:
        print('Creating 2D spectra...')
        omin = 63
        omax = 104
        rv = dir[len(basedir):-1]
        print(rv)
        while omin <= omax:
            fileout = dir + str(int(omin)) + '_2D.csv'
            print(fcam, rv, omin)
            if not os.path.exists(fileout):
                fileorder = open(fileout, 'w')
                fileorder.write('xpix,ypix,wave,flux\n')
                orddata = pd.read_csv(dir + str(int(omin)) + '.csv', sep=',')
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


def do_ccf_all(fcam):
    rvrange = np.arange(-10000, 10000, 250)

    for rv in rvrange:
        ccf.ccf2(rv, fcam)


def do_moes_full_spectrum(rv, fcam):
    #basedir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
    basedir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
    #outpath = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f'+str(int(fcam))+'mm/slit_files/'+str(int(rv))+'/'

    slitfiles = basedir+'/slit_files/'+str(int(rv))+'/'
    outpath = basedir + '/moes_files/' + str(int(rv)) + '/'

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    print('Creating MOES spectrum...')
    omin = 63
    omax = 104
    while omin <= omax:
        print(fcam, rv, omin)
        fileout = outpath + str(int(omin)) + '.tsv'
        data = pd.read_csv(slitfiles + str(int(omin)) + '_slit.tsv', sep=',')

        if not os.path.exists(fileout):
            # print(data)
            datord = data.loc[data['order'] == omin]
            specmoes = fideos_spectrograph.tracing_full_fcam(datord, fcam)
            specmoes.to_csv(outpath + str(int(omin)) + '.tsv', index=False)

        else:
            print('File already created...')

        omin += 1


def do_slit_files(rv, fcam):
    echelle_orders.init_stellar_doppler(rv, fcam)
    print('spectrum file written... rv = ', rv, 'fcam = ', fcam, '\n')


def do_slit_moes_all(fcam):
    rvs = np.arange(-10000, 10001, 250)
    print(rvs)
    #for rv in rvs:
    #    do_slit_files(rv, fcam)

    for rv in rvs:
        do_moes_full_spectrum(rv, fcam)


def do_2D(rv, fcam):
    moesdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/moes_files/'+str(int(rv))+'/'
    outdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/2D/'+str(int(rv))+'/'
    print('Creating 2D spectra...')
    omin = 63
    omax = 104
    print(rv)
    while omin <= omax:
        fileout = outdir + str(int(omin)) + '_2D.tsv'
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


def do_all_ccf(fcam):
    rvs = np.arange(-10000, 10001, 250)
    for rv in rvs:
        # do_2D(rv, fcam)
        ccf.ccf2(rv, fcam)


def do_ccf_fit_all(fcam):
    rvs = np.arange(-10000, -5001, 250)
    drv = []
    for rv in rvs:
        drv.append(float(ccf.ccf_voigt(rv, fcam)))

    n, bins, patch = plt.hist(drv, bins = 11)
    print(bins)
    height = max(n)
    mean = np.mean(bins)
    sigma = np.std(bins)
    rvmin = min(drv)
    rvmax = max(drv)
    rv_array = np.arange(rvmin, rvmax, 0.1)
    offset = 0.
    #popt, pcov = curve_fit(gaus, bins, n, p0=[height, mean, sigma]) #, offset])
    #plt.plot(rv_array, gaus(rv_array, *popt), 'r-')
    plt.xlabel(r'$\Delta$RV (m/s)')
    plt.ylabel('N')
    plt.show()


def plot_spectrum():

    outdir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(300)) + 'mm/2D/'+str(int(0))+'/'
    #tempix = pd.read_csv('stellar_spectrum/star_template_orders/' + str(80) + '_star_spec.dat', sep=',')
    #print(tempix)
    omin = 63
    omax = 104
    while omin < omax:
        #tempix = pd.read_csv('stellar_spectrum/star_template_orders/' + str(int(omin)) + '_star_spec.dat', sep=',')
        tempix = pd.read_csv(outdir+str(int(omin)) + '_2D.tsv', sep=',')
        plt.plot(tempix['xpix'], tempix['ypix'], 'k.')
        omin += 1

    plt.xlim(0, 2048)
    plt.ylim(0, 2048)
    plt.show()
    #    data = pd.read_csv(outdir+'80_2D.tsv', sep=',')
    #    plt.plot(data['wave'], data['flux'], 'k-')
    #    plt.show()


if __name__ == '__main__':
    #fcam = 230
    #create_rv_slit_files(fcam)
    #create_moes_spectrum(fcam)
    #create_2D_spectra(fcam)
    #do_2D(-8250, fcam)
    #do_moes_full_spectrum(-8250, fcam)
    #do_ccf_fit_all(300)
    do_all_ccf(230)
    #plot_spectrum()
    #plot_spectrum()
    #do_slit_moes_all(fcam)
