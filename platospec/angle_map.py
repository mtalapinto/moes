import math
import sys
import numpy as np
import os
import pandas as pd
from scipy import interpolate
import platospec_moes
import echelle_orders
import matplotlib.pyplot as plt


def create_spectrum(rv, det, order):
    instr = 'platospec'
    slitout = echelle_orders.init_line_doppler_full(rv, instr, order)
    print('Ray tracing with moes... '),
    specout = platospec_moes.tracing_full_det(slitout, det)
    print('done')
    waveout, fluxout, pixout, yout, sampout, pixsizeout = [], [], [], [], [], []
    pixtiltout, npixout = [], []
    specout = specout.loc[specout['x'] >= 0]
    specout = specout.loc[specout['x'] <= det[2]]
    specout = specout.loc[specout['y'] >= 0]
    specout = specout.loc[specout['y'] <= det[3]]
    x = 0.
    xmax = int(det[2])
    outdir = '/data/matala/moes/platospec/results/anglemaps/'  #ccd_'+str(det[-1])+'/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    xout, outtheta = [], []
    while x < xmax:
        pixdata = specout.loc[specout['x'] > x]
        pixdata = pixdata.loc[pixdata['x'] < x + 1]
        waves = np.unique(specout['wave'])
        angout = []
        for w in waves:
            line = pixdata.loc[pixdata['wave'] == w]
            if len(line) > 1:
                coef = np.polyfit(line['x'], line['y'], 1)
                linefunc = np.poly1d(coef)
                xarray = np.arange(min(line['x']) - 2, max(line['x']) + 2)
                angle = np.rad2deg(np.arctan2(max(linefunc(xarray)) - min(linefunc(xarray)), max(xarray) - min(xarray)))
                angleout = 90. - angle
                angout.append(angleout)
        
        outangle=np.median(angout)
        xout.append(x)
        outtheta.append(outangle)
        line = 'order = ' + str(order) + ', x = ' + str(x) + ', theta = ' + str(np.round(outangle, 5)) + ' deg, ' + str(int(x * 100 / xmax)) + '%'
        print(line, end="\r", flush=True)
        #print(order, x, outangle)
        x += 1.
    
    outfile = pd.DataFrame()
    outfile['x'] = xout
    outfile['theta'] = outtheta
    outfile.to_csv(outdir + str(order) + '.csv', index=False)



def do_map(n):
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(4.5, 21.1, 1.5)
    fcam = 240.
    fcol = 876.
    slit = 100
    detdir = "".join(['/data/matala/moes/platospec/results/anglemaps/ccd_' + str(int(n)) + '/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    omin = 68
    omax = 122
    rv = 0.
    #det = [2.02, 15., 2048., 2048., 5]
    i = 0
    for pixsize in pixarray:
        print('Pixel size = ', pixsize, 'index ', i, ', detector = ', n)
        samp = fcam / fcol * slit / pixsize
        x_pix = x_um / pixsize
        y_pix = y_um / pixsize
        det = [samp, pixsize, x_pix, y_pix, i]
        if int(i) == int(n):
            while omin <= omax:
                print(omin)
                anglefile = detdir + str(int(omin)) + '.csv'
                if not os.path.exists(anglefile):
                    create_spectrum(rv, det, omin)
                else:
                    print('File already created!')
                omin += 1
        i += 1


    #plt.show()


if __name__ == '__main__':
    do_map(sys.argv[-1])
    #det = [2.02, 15., 2048., 2048., 6]
    #rv = 0
    #omin = 122
    #create_spectrum(rv, det, omin)
        


