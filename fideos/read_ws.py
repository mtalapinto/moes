import os.path
import pickle as pkl
import pandas as pd
import glob
import matplotlib.pyplot as plt
from astropy.time import Time


def get_ws():
    basedir = '/luthien/fideos/data/ws/'
    wsfiles = glob.glob(basedir+'*ThAr*')
    plotoutdir = 'plots/ws/'
    if not os.path.exists(plotoutdir):
        os.mkdir(plotoutdir)

    #print(wsfiles)

    for i in range(len(wsfiles)):
        date_ws = wsfiles[i][len(basedir):-27].replace('_', ':')
        jd = Time(date_ws, format='isot').jd
        date_ws = date_ws[:10]
        outdir = "".join([basedir, date_ws, '/'])
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fib = wsfiles[i][(len(basedir)+36):-15]
        print(fib)
        outfile = "".join([outdir, 'ws_',str(fib), '_', str(i), '.pkl'])
        ws = pd.read_pickle(wsfiles[i])
        plt.plot(ws['All_Pixel_Centers_co'], ws['All_Wavelengths_co'], 'k.')
        plt.savefig(plotoutdir + str(date_ws) + '_' + str(fib)+'_ws_'+str(i)+'.png')
        plt.close()
        outdata = {'jd': jd, 'x': ws['All_Pixel_Centers_co'], 'wave': ws['All_Wavelengths_co']}
        outwrite = open(outfile, 'wb')
        pkl.dump(outdata, outwrite)
        outwrite.close()

        if i == 3:
            print(date_ws, jd)
            print(wsfiles[i])
            ws = pd.read_pickle(wsfiles[i])
            trace = pd.read_pickle(basedir+'trace.pkl')
            print(ws.keys())
            print(trace.keys())
            print(len(trace['c_ob']))
            #print(trace['c_co'])
            #print(trace['nord_ob'])
            #print(trace['nord_co'])
            #print(len(ws['rms_ms_co']))
            #print(len(ws['G_ord_co']))
            #print(len(ws['G_pix_co']))
            #print(len(ws['All_Pixel_Centers_co']))
            #print(len(ws['All_Wavelengths_co']))
            #print(ws['mjd'])
            #print(len(ws['II_co']))
            #print(len(ws['G_res_co']))
            #print(len(ws['G_wav_co']))
            #print(ws['All_Pixel_Centers_co'])
            plt.plot(ws['All_Pixel_Centers_co'], ws['All_Wavelengths_co'], 'k.')
            plt.show()
            #plt.clf()
            #plt.close()



if __name__ == '__main__':

    get_ws()