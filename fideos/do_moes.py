import os
import pandas as pd
import fideos_spectrograph
import numpy as np


def do_moes_full_spectrum(rv, fcam):
    #basedir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
    #basedir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
    basedir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
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


def do_moes_order(rv, fcam, omin):
    #basedir = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
    basedir = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/'
    slitfiles = basedir+'/slit_files/'+str(int(rv))+'/'
    outpath = basedir + '/moes_files/' + str(int(rv)) + '/'

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


def do_moes_all(fcam):
    rvs = np.arange(9750, 10001, 250)
    for rv in rvs:
        do_moes_full_spectrum(rv, fcam)


if __name__ == '__main__':

    #do_moes_all(230)
    do_moes_full_spectrum(0, 300)
    #do_moes_order(-6750, 230, 94)
    import time

    #time.sleep(10)
