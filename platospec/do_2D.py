import math
import sys
import numpy as np
import os
import pandas as pd
from scipy import interpolate
import platospec_moes
import echelle_orders
import matplotlib.pyplot as plt


def get_tilt_angle_at_pixel(waves, spec, n):
    # we compute tilt by fitting a line
    angleout = 0.
    line0 = spec.loc[spec['wave'] == waves[np.random.randint(0, len(waves) - 1)]]
    line0_sort = line0.sort_values(by=['x'])
    if len(line0) == int(n):
        hitail = line0_sort.tail(7)
        lotail = line0_sort.head(7)

        coef = np.polyfit(hitail['x'], hitail['y'], 1)
        coeflo = np.polyfit(lotail['x'], lotail['y'], 1)
        poly1d_fn = np.poly1d(coef)
        poly1d_fn2 = np.poly1d(coeflo)
        xarray = np.arange(min(line0['x']) - 2, max(line0['x']) + 2)
        angle_lo = np.rad2deg(
            np.arctan2(max(poly1d_fn(xarray)) - min(poly1d_fn(xarray)), max(xarray) - min(xarray)))
        angle_hi = np.rad2deg(
            np.arctan2(max(poly1d_fn2(xarray)) - min(poly1d_fn2(xarray)), max(xarray) - min(xarray)))
        angle = (angle_lo + angle_hi) / 2
        angleout = 90. - angle

    else:
        angleout = 0.

    return angleout


def is_between_lines(point, line1, line2):
    """
    Check if a point is between two lines defined by line1 and line2.
    """
    x, y = point
    x1, y1 = line1
    x2, y2 = line2
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def select_points_between_lines(points, line1, line2):
    """
    Select all points from the input list that are between two lines defined by line1 and line2.
    """
    selected_points = [point for point in points if is_between_lines(point, line1, line2)]
    return selected_points


def do_2D(rv, fcam):
    # moesdir = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/moes/'+str(int(rv))+'/'
    moesdir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/moes/' + str(int(rv)) + '/'
    outdir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/2D/' + str(int(rv)) + '/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print('Creating 2D spectra...')
    omin = 73
    omax = 114
    print(rv)
    while omin <= omax:
        fileout = outdir + str(int(omin)) + '_2D.tsv'
        print(fcam, rv, omin)
        if not os.path.exists(fileout):
            fileorder = open(fileout, 'w')
            fileorder.write('ypix,xpix,wave,flux\n')
            orddata = pd.read_csv(moesdir + str(int(omin)) + '.tsv', sep=',')

            orddata = orddata.loc[orddata['x'] <= 2048]
            orddata = orddata.loc[orddata['x'] >= 0]
            orddata = orddata.loc[orddata['y'] <= 2048]
            orddata = orddata.loc[orddata['y'] >= 0]
            ymin = np.min(orddata['x'].values) + 1
            ymax = np.max(orddata['x'].values) - 1
            yini = int(ymin)
            yend = int(ymax)

            while yini <= yend:
                data_per_ypix = orddata.loc[orddata['x'] > yini - 0.5]
                data_per_ypix = data_per_ypix.loc[data_per_ypix['x'] <= yini + 0.5]
                xline = np.mean(data_per_ypix['y'].values)
                yline = yini
                waveline = np.mean(data_per_ypix['wave'].values)
                fluxline = np.mean(data_per_ypix['flux'].values)
                fileorder.write('%f,%f,%f,%f\n' % (float(xline), float(yline), float(waveline), float(fluxline)))
                yini += 1
            fileorder.close()

        else:
            print('File already created...')
        omin += 1


def do_2D_per_order(rv, det, orddata, instr):
    # moesdir = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/moes/'+str(int(rv))+'/'
    # moesdir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/moes/' + str(int(rv)) + '/'
    # outdir = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/2D/'+str(int(rv))+'/'
    detdir = "".join(['data/pix_exp/'])
    outdir = "".join([detdir, 'ccd_', str(int(det[-1])), '/'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    rvdir = outdir + str(int(rv)) + '/'
    if not os.path.exists(rvdir):
        os.mkdir(rvdir)

    x, y, waveout, flux = [], [], [], []
    omin = orddata['order'].values[0]
    waves = np.unique(orddata['wave'])

    fileout = rvdir + str(int(omin)) + '_2D.tsv'
    print('Detector número ', det[-1])
    print('Pixel size = ', det[0], ', ', det[1], 'x', det[2], ' pixels, ', 'RV = ', rv, ', Order =  ', omin)
    if not os.path.exists(fileout):
        # orddata = orddata.loc[orddata['x'] <= 2048]
        # orddata = orddata.loc[orddata['x'] >= 0]
        # orddata = orddata.loc[orddata['y'] <= 2048]
        # orddata = orddata.loc[orddata['y'] >= 0]
        ymin = np.min(orddata['x'].values) + 1
        ymax = np.max(orddata['x'].values) - 1
        yini = int(ymin)
        yend = int(ymax)

        while yini <= yend:
            data_per_ypix = orddata.loc[orddata['x'] > yini - 0.5]
            data_per_ypix = data_per_ypix.loc[data_per_ypix['x'] <= yini + 0.5]
            xline = np.mean(data_per_ypix['y'].values)
            yline = yini
            waveline = np.mean(data_per_ypix['wave'].values)
            fluxline = np.mean(data_per_ypix['flux'].values)
            x.append(xline)
            y.append(yline)
            waveout.append(waveline)
            flux.append(fluxline)
            yini += 1

        datawrite = pd.DataFrame()
        datawrite['x'] = x
        datawrite['y'] = y
        datawrite['wave'] = waveout
        datawrite['flux'] = flux
        datawrite.to_csv(fileout, index=False)

    else:
        print('File already created...')


def create_2D_moes_spectrum(rv, det, instr, order):
    # platospec omin = 73, omax = 114, fideos omin = 63, omax = 104
    slitout = echelle_orders.init_stellar_doppler(rv, instr, order)
    # Doing Moe's ray tracing
    specout = platospec_moes.tracing_full_det(slitout, det)
    print('Moes spectrum created...')
    print('Creating 2D spectrum')
    do_2D_per_order(rv, det, specout, instr)
    print('order ', order, 'created')

    # import matplotlib.pyplot as plt
    # plt.plot(specout['x'], specout['y'], 'r.')
    # plt.show()


def create_2D_moes_simple_spec(rv, det, instr, order):
    print('Creating spectrum for RV = ', rv, ', order = ', order, ', samp = ', det[0])
    slitout = echelle_orders.init_stellar_doppler_simple(rv, instr, order)
    # print(slitout)
    print('Detector number = ', det[-1])
    detdir = 'data/pix_exp/ns0/ccd_' + str(det[-1]) + '/'
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    outdir = 'data/pix_exp/ns0/ccd_' + str(det[-1]) + '/' + str(int(rv)) + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Doing Moe's ray tracing
    print('Ray tracing with moes... ', )
    specout = platospec_moes.tracing_full_det(slitout, det)
    print('done')
    specout = specout.loc[specout['x'] >= 0]
    specout = specout.loc[specout['x'] <= det[2]]
    specout = specout.loc[specout['y'] >= 0]
    specout = specout.loc[specout['y'] <= det[3]]
    # plt.plot(specout['x'], specout['y'], 'k.')
    # plt.show()
    # print(specout)
    wmin = min(specout['wave'].values) * 1e4
    wmax = max(specout['wave'].values) * 1e4
    pini = int(min(specout['x'].values))
    pend = int(max(specout['x'].values))
    waux = wmin
    waveout, fluxout, pixout, sampout = [], [], [], []
    print('Creating 1D spectra...')
    while pini <= pend:
        pixdata = specout.loc[specout['x'] >= pini]
        pixdata = pixdata.loc[pixdata['x'] <= pini + 1]

        if math.isnan(np.mean(pixdata['flux'].values)):
            print(pini, pend)
            print(pixdata)
        waveout.append(np.mean(pixdata['wave']))
        fluxout.append(np.mean(pixdata['flux']))
        pixout.append(pini)
        sampout.append(det[0])
        pini += 1

    dataout = pd.DataFrame()
    dataout['wave'] = waveout
    dataout['flux'] = fluxout
    dataout['pix'] = pixout
    dataout['samp'] = sampout
    print('Saving .tsv file...', )
    dataout.to_csv(outdir + str(order) + '_2D_moes.tsv', sep=',', index=False)
    print('done')

    # else:
    #    print('Spectrum already created.')
    # plt.plot(dataout['wave']*1e4, dataout['flux'], 'r-', alpha=0.5)
    # tempdata = pd.read_csv('stellar_template/stellar_template.tsv', sep=',')
    # plt.plot(tempdata['wave'], tempdata['flux'],'k-', alpha=0.5)
    # plt.show()


def create_2D_moes_full_spec(rv, det, instr, order):
    print('Creating spectrum for RV = ', rv, ', order = ', order, ', samp = ', det[0])
    slitout = echelle_orders.init_stellar_doppler_full(rv, instr, order)
    npoints = len(slitout.loc[slitout['wave'] == slitout['wave'].values[np.random.randint(0, len(slitout) - 1)]])

    # print(slitout)
    print('Detector number = ', det[-1])
    detdir = '/data/matala/luthien/platospec/data/pix_exp/ns0_full/ccd_' + str(int(det[-1])) + '/'
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    outdir = detdir + str(int(rv)) + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Doing Moe's ray tracing
    print('Ray tracing with moes... '),
    specout = platospec_moes.tracing_full_det(slitout, det)
    print('done')
    waveout, fluxout, pixout, yout, sampout, pixsizeout = [], [], [], [], [], []
    pixtiltout, npixout = [], []
    specout = specout.loc[specout['x'] >= 0]
    specout = specout.loc[specout['x'] <= det[2]]
    specout = specout.loc[specout['y'] >= 0]
    specout = specout.loc[specout['y'] <= det[3]]

    xmin = np.min(specout['x'].values) + 1
    xmax = np.max(specout['x'].values) - 1
    xini = int(xmin)
    xend = int(xmax)
    icount = 0.
    print('Creating 1D spectra...')
    maxicount = np.abs(xend - xini)
    while xini <= xend:
        data_per_ypix = specout.loc[specout['x'] > xini - 0.5]
        data_per_ypix = data_per_ypix.loc[data_per_ypix['x'] <= xini + 0.5]
        wavesinpix = data_per_ypix['wave'].values
        pixtilt = get_tilt_angle_at_pixel(wavesinpix, specout, npoints)
        xline = xini
        yline = np.mean(data_per_ypix['y'].values)
        waveline = np.mean(data_per_ypix['wave'].values)
        fluxline = np.mean(data_per_ypix['flux'].values)
        waveout.append(waveline)
        fluxout.append(fluxline)
        pixout.append(xline)
        pixtiltout.append(pixtilt)
        yout.append(yline)
        sampout.append(det[0])
        pixsizeout.append(det[1])
        npixout.append(det[2])
        print(str(int(icount * 100 / maxicount)), '%', end="\r", flush=True)
        icount += 1
        xini += 1

    dataout = pd.DataFrame()
    dataout['wave'] = waveout
    dataout['flux'] = fluxout
    dataout['pix'] = pixout
    dataout['y'] = yout
    dataout['samp'] = sampout
    dataout['pixsize'] = pixsizeout
    dataout['pixtilt'] = pixtiltout
    print('Saving .tsv file...'),
    dataout.to_csv(outdir + str(order) + '_2D_moes.tsv', sep=',', index=False)
    print('done')

    # else:
    #    print('Spectrum already created.')
    # plt.plot(dataout['wave']*1e4, dataout['flux'], 'r-', alpha=0.5)
    # tempdata = pd.read_csv('stellar_template/stellar_template.tsv', sep=',')
    # plt.plot(tempdata['wave'], tempdata['flux'],'k-', alpha=0.5)
    # plt.show()


def create_2D_moes_tilt_spec(rv, det, instr, order):

    print('Creating spectrum for RV = ', rv, ', order = ', order, ', samp = ', det[0])
    slitout = echelle_orders.init_stellar_doppler_full(rv, instr, order)
    npoints = len(slitout.loc[slitout['wave'] == slitout['wave'].values[np.random.randint(0, len(slitout) - 1)]])

    # print(slitout)
    print('Detector number = ', det[-1])
    detdir = '/data/matala/luthien/platospec/data/pix_exp/ns0_tilt/ccd_' + str(int(det[-1])) + '/'
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    outdir = detdir + str(int(rv)) + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Doing Moe's ray tracing
    print('Ray tracing with moes... '),
    specout = platospec_moes.tracing_full_det(slitout, det)
    print('done')

    waveout, fluxout, pixout, yout, sampout, pixsizeout = [], [], [], [], [], []
    pixtiltout, npixout = [], []
    specout = specout.loc[specout['x'] >= 0]
    specout = specout.loc[specout['x'] <= det[2]]
    specout = specout.loc[specout['y'] >= 0]
    specout = specout.loc[specout['y'] <= det[3]]

    #wave0 = specout['wave'].values[4500]
    #wave0data = specout.loc[specout['wave'] == wave0]
    #plt.plot(wave0data['x'], wave0data['y'], 'ko')
    #plt.show()
    #plt.clf()
    #plt.close()
    angles = pd.read_csv('/data/matala/moes/platospec/results/anglemaps/ccd_' + str(int(det[-1]))+ '/' + str(order) + '.csv')

    xmin = np.min(specout['x'].values) + 5   # remove first pixels
    xmax = np.max(specout['x'].values) - 5   # remove last pixels
    xini = int(xmin)
    xend = int(xmax)
    icount = 0.
    print('Creating 1D spectra...')
    maxicount = np.abs(xend - xini)
    while xini <= xend:
        thetadata = angles.loc[angles['x'] == float(xini)]
        theta = thetadata['theta'].values
        #print(theta)
        pix_low_edge = xini - 0.5
        pix_hi_edge = xini + 0.5

        data_per_ypix = specout.loc[specout['x'] > xini - 0.5]
        data_per_ypix = data_per_ypix.loc[data_per_ypix['x'] <= xini + 0.5]

        y_min = min(data_per_ypix['y'])
        #print(y_min)
        y_max = y_min + 30.
        yarr_lo = np.arange(y_min, y_max, 0.1)
        if len(yarr_lo) > 0:
            #print(len(yarr_lo), theta, y_min, pix_low_edge)
            xarr_lo = np.tan(theta * np.pi / 180) * (yarr_lo - y_min) + pix_low_edge
            yarr_hi = np.arange(y_min, y_max, 0.1)
            xarr_hi = np.tan(theta * np.pi / 180) * (yarr_hi - y_min) + pix_hi_edge

            specoutaux = specout.loc[specout['x'] <= max(xarr_hi)]
            specoutaux = specoutaux.loc[specoutaux['x'] >= min(xarr_lo)]
            specoutaux = specoutaux.loc[specoutaux['y'] <= max(yarr_hi)]
            specoutaux = specoutaux.loc[specoutaux['y'] >= min(yarr_lo)]

            m1 = 1 / np.tan(theta * np.pi / 180)
            m2 = 1 / np.tan(theta * np.pi / 180)
            linelo = m1 * (xarr_lo - pix_low_edge) + y_min
            linehi = m2 * (xarr_hi - pix_hi_edge) + y_min

            # plt.plot(xarr_lo, yarr_lo, 'k-')
            # plt.plot(xarr_hi, yarr_hi, 'k-')

            # plt.plot(xarr_lo, linelo, 'r--', zorder=10)
            # plt.plot(xarr_hi, linehi, 'r--', zorder=10)
            # plt.plot(specoutaux['x'], specoutaux['y'], 'bo')
            # print(len(specoutaux))
            # sub_df = df[((df['A'] < 0.4) & (df['B'] > 0.4)) | (df['B'] < 0.4) & (df['A'] < (0.2 + 0.25 * df['B']))]
            tiltpixdata = specoutaux.loc[(specoutaux['y'] < m1 * (specoutaux['x'] - pix_low_edge) + y_min)]
            tiltpixdata = tiltpixdata.loc[(tiltpixdata['y'] > m2 * (tiltpixdata['x'] - pix_hi_edge) + y_min)]

            # query_expression = f"({m1} * (x - {pix_low_edge}) + {y_min}) <= y <= ({m2} * (x - {pix_hi_edge}) + {y_min})"
            # result_df = specoutaux.query(query_expression)
            # print('Orinting pix tilt data')
            # print(tiltpixdata)
            # plt.plot(tiltpixdata['x'], tiltpixdata['y'], 'ro', zorder=5)
            # condition = (specoutaux['x'] >= m1 * (specoutaux['x'].values - pix_low_edge) + y_min) & (
            #            specoutaux['y'] <= m2 * (specoutaux['x'].values - pix_hi_edge) + y_min)
            # data_per_stripe = specout[condition].reset_index(drop=True)
            # print(data_per_stripe)

            # plt.show()
            # plt.clf()
            # plt.close()

            wavesinpix = tiltpixdata['wave'].values
            xline = xini
            yline = np.mean(tiltpixdata['y'].values)
            waveline = np.mean(tiltpixdata['wave'].values)
            fluxline = np.mean(tiltpixdata['flux'].values)
            waveout.append(waveline)
            fluxout.append(fluxline)
            pixout.append(xline)
            pixtiltout.append(theta)
            yout.append(yline)
            sampout.append(det[0])
            pixsizeout.append(det[1])
            npixout.append(det[2])

        print(str(int(icount * 100 / maxicount)), '%', end="\r", flush=True)
        icount += 1
        xini += 1

    dataout = pd.DataFrame()
    dataout['wave'] = waveout
    dataout['flux'] = fluxout
    dataout['pix'] = pixout
    dataout['y'] = yout
    dataout['samp'] = sampout
    dataout['pixsize'] = pixsizeout
    dataout['pixtilt'] = pixtiltout
    print('Saving .tsv file...'),
    dataout.to_csv(outdir + str(order) + '_2D_moes.tsv', sep=',', index=False)
    print('done')

    # else:
    #    print('Spectrum already created.')
    # plt.plot(dataout['wave']*1e4, dataout['flux'], 'r-', alpha=0.5)
    # tempdata = pd.read_csv('stellar_template/stellar_template.tsv', sep=',')
    # plt.plot(tempdata['wave'], tempdata['flux'],'k-', alpha=0.5)
    # plt.show()


def create_spectra(instr, fcam):
    wmin = 0.38
    wmax = 0.68
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    print('Creating MOES spectra for ' + str(instr))
    # rvarray = np.arange(-10000, 10001, 250)
    rvarray = [-9500]
    # rvarray = [0]
    for rv in rvarray:
        ominaux = omin
        omaxaux = omax
        while ominaux <= omaxaux:
            create_2D_moes_spectrum(rv, fcam, instr, ominaux)
            ominaux += 1


def create_spectra_det(instr, det):
    wmin = 0.38
    wmax = 0.68
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    print('Creating MOES spectra for ' + str(instr))
    if det[-1] == 0:
        rvarray = np.arange(10000, 10001, 250)
    elif det[-1] == 1:
        rvarray = np.arange(10000, 10001, 250)
    elif det[-1] == 2:
        rvarray = np.arange(0, 10001, 250)
    else:
        rvarray = np.arange(-10000, 10001, 250)
    # rvarray = [-9500]
    # rvarray = [0]
    for rv in rvarray:
        ominaux = omin
        omaxaux = omax
        # omaxaux = omax
        while ominaux <= omaxaux:
            create_2D_moes_spectrum(rv, det, instr, ominaux)
            ominaux += 1


def create_spectra_rv(instr, det, rv):
    wmin = 0.38
    wmax = 0.68
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876
        owmin = int(2 * np.sin(blaze_angle) / (G * wmin))
        owmax = int(2 * np.sin(blaze_angle) / (G * wmax))
        omax = owmin
        omin = owmax

    print('Creating MOES spectra for ' + str(instr))
    rvarray = [rv]
    for rv in rvarray:
        ominaux = omin
        omaxaux = omax
        # omaxaux = omax
        while ominaux <= omaxaux:
            create_2D_moes_spectrum(rv, det, instr, ominaux)
            ominaux += 1


def do_all_data():
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(7.5, 18, 1.5)
    fcam = 240.
    fcol = 876.
    slit = 100
    i = 0
    detdir = "".join(['data/pix_exp/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    logfile = open(detdir + 'pix_experiment_logfile.txt', 'w')
    logfile.write('ccd,xpix,ypix,pixsize\n')
    for pixsize in pixarray:
        samp = fcam / fcol * slit / pixsize
        x_pix = x_um / pixsize
        y_pix = y_um / pixsize
        # print(samp, x_pix, y_pix)
        det = [samp, pixsize, x_pix, y_pix, i]
        # logfile.write('%f,%f,%f,%f\n' %(i, x_pix, y_pix, pixsize))
        print(det)
        # if i == 6:
        # create_spectra_det('platospec', det)
        i += 1


def do_single_dataset(rv, detno):
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(7.5, 18, 1.5)
    fcam = 240.
    fcol = 876.
    slit = 100
    i = 0
    detdir = "".join(['data/pix_exp/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)

    x_pix = x_um / pixarray[detno]
    y_pix = y_um / pixarray[detno]
    det = [pixarray[detno], x_pix, y_pix, detno]
    create_spectra_rv('platospec', det, rv)


def do_all_2D_moes_simple(n):
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(4.5, 21.1, 1.5)
    # pixarray = [6.5]
    fcam = 240.
    fcol = 876.
    slit = 100
    detdir = "".join(['data/pix_exp/ns0/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    rvs = np.arange(-10000, 10001, 100)
    # rvs = [-10000]
    i = 0
    for pixsize in pixarray:
        print(pixsize)
        for rv in rvs:
            samp = fcam / fcol * slit / pixsize
            x_pix = x_um / pixsize
            y_pix = y_um / pixsize
            det = [samp, pixsize, x_pix, y_pix, i]
            # print(i, n)
            if i == n:
                omin = 68
                omax = 122
                print(i, n)
                while omin <= omax:
                    create_2D_moes_simple_spec(rv, det, 'platospec', omin)
                    omin += 1
        i += 1


def do_all_2D_moes_full(n):
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(4.5, 21.1, 1.5)
    # pixarray = [4.5]
    fcam = 240.
    fcol = 876.
    slit = 100
    detdir = "".join(['/data/matala/luthien/platospec/data/pix_exp/ns0_full/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    rvs = np.arange(-10000, 10001, 100)
    # rvs = [-10000]
    i = 0
    for pixsize in pixarray:
        print('Pixel size = ', pixsize)
        for rv in rvs:
            samp = fcam / fcol * slit / pixsize
            x_pix = x_um / pixsize
            y_pix = y_um / pixsize
            det = [samp, pixsize, x_pix, y_pix, i]
            # print(i, n)
            if i == n:
                omin = 68
                omax = 122
                print('Detector no.', n)
                while omin <= omax:
                    # create_2D_moes_simple_spec(rv, det, 'platospec', omin)
                    # CHECK IF FILE EXIST
                    filename = detdir + 'ccd_' + str(int(n)) + '/' + str(rv) + '/' + str(omin) + '_2D_moes.tsv'
                    print(filename)
                    if os.path.exists(filename):
                        print('File already created...\n')
                    else:
                        create_2D_moes_full_spec(rv, det, 'platospec', omin)
                    omin += 1
        i += 1


def do_all_2D_moes_tilt(n):
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(4.5, 21.1, 1.5)
    # pixarray = [4.5]
    fcam = 240.
    fcol = 876.
    slit = 100
    detdir = "".join(['/data/matala/luthien/platospec/data/pix_exp/ns0_tilt/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    rvs = np.arange(5000, 10001, 100)
    # rvs = [-10000]
    i = 0
    for pixsize in pixarray:
        print('Pixel size = ', pixsize)
        for rv in rvs:
            samp = fcam / fcol * slit / pixsize
            x_pix = x_um / pixsize
            y_pix = y_um / pixsize
            det = [samp, pixsize, x_pix, y_pix, i]
            #print(pixsize, i, n)
            if i == n:
                omin = 68
                omax = 122
                print('Detector no.', n, ' Pixel size = ', pixsize)
                while omin <= omax:
                    # create_2D_moes_simple_spec(rv, det, 'platospec', omin)
                    # CHECK IF FILE EXIST
                    filename = detdir + 'ccd_' + str(int(n)) + '/' + str(rv) + '/' + str(omin) + '_2D_moes.tsv'
                    print(filename)
                    if os.path.exists(filename):
                        print('File already created...\n')
                    else:
                        print('Creating tilt based spectrum...'),
                        create_2D_moes_tilt_spec(rv, det, 'platospec', omin)
                        print('Done')
                    omin += 1
        i += 1


def sampling(detno):
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixarray = np.arange(7.5, 18, 1.5)
    fcam = 240.
    fcol = 876.
    slit = 100
    slit_ccd = slit * fcam / fcol

    i = 0
    detdir = "".join(['data/pix_exp/'])
    if not os.path.exists(detdir):
        os.mkdir(detdir)

    x_pix = x_um / pixarray[detno]
    y_pix = y_um / pixarray[detno]
    det = [pixarray[detno], x_pix, y_pix, detno]
    samp = slit_ccd / pixarray[detno]
    return samp


def create_noisy_data(det, rv, order, sn):
    print('Creating noisy data')
    print(det, rv, order)

    # stardata = pd.read_csv('stellar_template/stellar_template_v2.tsv', sep=' ')
    # wini = 3800
    # wend = 6800
    basedir = 'data/pix_exp/'
    basesndir = 'data/pix_exp/ns0/'
    sndir = basedir + 'ns' + str(int(sn * 100)) + '/'
    if not os.path.exists(sndir):
        os.mkdir(sndir)
    detdir = sndir + 'ccd_' + str(det) + '/'
    if not os.path.exists(detdir):
        os.mkdir(detdir)
    rvdir = detdir + str(rv) + '/'
    if not os.path.exists(rvdir):
        os.mkdir(rvdir)

    data = pd.read_csv(basesndir + 'ccd_' + str(det) + '/' + str(rv) + '/' + str(order) + '_2D_moes.tsv', sep=',')
    # wini = min(data['wave'].values)
    # wend = max(data['wave'].values)
    # stardata = stardata.loc[stardata['WAVE'] <= wend*1e4]
    # stardata = stardata.loc[stardata['WAVE'] >= wini*1e4]
    # snr_temp = DER_SNR(stardata['FLUX'])
    # print(snr_temp)
    # we add noise of 5%
    # snr_i = DER_SNR(data['flux'].values)
    # print(snr_i)
    ns = np.random.normal(0, 1 * sn, len(data))
    data['flux_ns'] = data['flux'] + ns

    dataout = pd.DataFrame()
    dataout['wave'] = data['wave']
    dataout['flux'] = data['flux_ns']
    dataout['pix'] = data['pix']
    dataout['samp'] = data['samp']
    dataout.to_csv(rvdir + str(order) + '_2D_moes.tsv', index=False, sep=',')
    # snr_f = DER_SNR(data['flux_ns'].values)
    # print(snr_f)
    # plt.figure(figsize=[8, 3])
    # plt.plot(data['wave'], data['flux'], 'k-', alpha=0.5)
    # plt.plot(data['wave'], data['flux_ns'], 'b-', alpha=0.5)
    # plt.show()
    # signal = np.median(data['flux'].values)
    # noise = np.sqrt(signal)
    # snr_ori = signal/noise
    # signal_new = np.median(data['flux_ns'].values)
    # noise_new = np.sqrt(signal_new)
    # snr_new = signal_new / noise_new
    # print(snr_ori, snr_new)


def create_all_noisy_data():
    rvs = np.arange(-10000, 10001, 100)
    dets = np.arange(0, 10, 1)
    omax = 122
    ns = 0.1  # 10%
    for det in dets:
        for rv in rvs:
            omin = 68
            while omin <= omax:
                create_noisy_data(det, rv, omin, ns)
                omin += 1


def DER_SNR(flux):
    # =====================================================================================
    """
    DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
                definition set forth by the Spectral Container Working Group of ST-ECF,
            MAST and CADC.

                signal = median(flux)
                noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
            snr    = signal / noise
                values with padded zeros are skipped

    USAGE       snr = DER_SNR(flux)
    PARAMETERS  none
    INPUT       flux (the computation is unit independent)
    OUTPUT      the estimated signal-to-noise ratio [dimensionless]
    USES        numpy
    NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum
            as a whole as long as
                * the noise is uncorrelated in wavelength bins spaced two pixels apart
                * the noise is Normal distributed
                * for large wavelength regions, the signal over the scale of 5 or
              more pixels can be approximated by a straight line

                For most spectra, these conditions are met.

    REFERENCES  * ST-ECF Newsletter, Issue #42:
                www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
                * Software:
            www.stecf.org/software/ASTROsoft/DER_SNR/
    AUTHOR      Felix Stoehr, ST-ECF
                24.05.2007, fst, initial import
                01.01.2007, fst, added more help text
                28.04.2010, fst, return value is a float now instead of a numpy.float64
    """
    from numpy import array, where, median, abs

    flux = array(flux)

    # Values that are exactly zero (padded) are skipped
    flux = array(flux[where(flux != 0.0)])
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if (n > 4):
        signal = median(flux)

        noise = 0.6052697 * median(abs(2.0 * flux[2:n - 2] - flux[0:n - 4] - flux[4:n]))

        return float(signal / noise)

    else:

        return 0.0


if __name__ == '__main__':
    # plottest(360)
    # do_2D_all(230)
    # do_2D(0, 360)
    # fcam = 300
    # do_single_dataset(-6750, 0)
    # plt.show()
    # do_all_data()
    # create_spectra('platospec', 240)
    # create_2D_moes_spectrum(0, 360, 'platospec')
    # pixelize_2D(0, 360)
    #do_all_2D_moes_tilt(float(sys.argv[-1]))
    do_all_2D_moes_tilt(float(sys.argv[-1]))
    # do_all_2D_moes_full(float(sys.argv[-1]))
    # create_all_noisy_data()
    # print(sampling(0))
    # do_2D_order(-6750, 230, 95)
    # import ti
