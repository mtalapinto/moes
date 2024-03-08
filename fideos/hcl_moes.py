import pandas as pd
import numpy as np
import fideos_spectrograph
import echelle_orders
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)


def th_spec():
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0
    thetamin = 0.
    thetamax = 360.
    while daux <= diam:
        while thetamin <= thetamax:
            if thetamin < 90:
                x = daux * np.cos(thetamin * np.pi / 180) - diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            elif 90 <= thetamin < 180:
                x = daux * np.cos(thetamin * np.pi / 180) - diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            elif 180 <= thetamin < 270:
                x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            else:
                x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                slitgrid.append(np.array([x, y]))

            thetamin += 15.
        thetamin = 0.
        daux += 0.025
    slitgrid = np.array(slitgrid)

    basedir = '/home/eduspec/Documentos/moes/fideos_moes/hcl_spectrum/'
    thdata = pd.read_csv(basedir+'th_lines.tsv', delim_whitespace=True)
    thdata['wave'] = 1.e4/thdata['sigma']  # in um

    wav_lo = 0.4  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    #print(ord_blu, ord_red)
    spectrum = []
    while ord_red < ord_blu:
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red)
        wav_max = wav_blz + wav_blz / (2 * ord_red)
        #print(wav_min, wav_max)
        ordlines = thdata.loc[thdata['wave'] < wav_max]
        ordlines = ordlines.loc[ordlines['wave'] > wav_min]
        #print(ordlines.columns)
        for i in range(len(ordlines)):
            for k in range(len(slitgrid)):
                H = np.zeros([3])
                H[0] = slitgrid[k][0]
                H[1] = slitgrid[k][1]
                DC = np.zeros([3])
                DC[2] = -1.
                single_element = (ord_red, ordlines['wave'].values[i], ordlines['int'].values[i], H[0], H[1], H[2], DC[0], DC[1], DC[2])
                spectrum.append(np.array(single_element))
        ord_red += 1

    spectrum = pd.DataFrame(spectrum, columns=['order', 'wave', 'flux', 'x', 'y', 'z', 'dx', 'dy', 'dz'])
    spectrum.to_csv(basedir+'th_spec.csv', index=False)


def th_moes():
    basedir = '/home/eduspec/Documentos/moes/fideos_moes/hcl_spectrum/'
    spectrum = pd.read_csv(basedir+'th_spec.csv',sep=',')
    specmoes = fideos_spectrograph.tracing_full_fcam(spectrum, 230)
    specmoes.to_csv(basedir+'th_spec_moes_230.csv', index=False)


def spec2D():
    stardata = pd.read_csv('hcl_spectrum/th_spec_moes.csv', sep=',')
    stardata = stardata.loc[stardata['x'] <= 2048]
    stardata = stardata.loc[stardata['x'] >= 0]
    stardata = stardata.loc[stardata['y'] <= 2048]
    stardata = stardata.loc[stardata['y'] >= 0]
    plt.plot(stardata['x'], stardata['y'],'k.')
    plt.show()
    orders = np.unique(stardata['order'].values)
    omin = min(orders)+1
    omax = max(orders)

    while omin <= omin + 1:
        print(omin)
        data = stardata.loc[stardata['order'].values == omin]
        wsdata = pd.read_csv('stellar_spectrum/star_template_orders/'+str(int(omin))+'_star_spec.dat', sep=',')
        ymin = np.min(data['y'].values) + 1
        ymax = np.max(data['y'].values) - 1
        yini = int(ymin)
        yend = int(ymax)
        fileorder = open('hcl_spectrum/th_spec/'+str(int(omin))+'.dat', 'w')
        fileorder.write('xpix,ypix,wave,flux\n')
        while yini <= yend:
            data_per_ypix = data.loc[data['y'] > yini - 0.5]
            data_per_ypix = data_per_ypix.loc[data_per_ypix['y'] <= yini + 0.5]
            #print(len(data_per_ypix))
            if len(data_per_ypix) == 0:
                wavedata = wsdata.loc[wsdata['ypix'] == yini]
                wave = wavedata['wave'].values
                xpix = wavedata['xpix'].values
                fileorder.write('%f,%f,%f,%f\n' % (float(xpix), float(yini), float(wave), float(0.)))
            else:
                xline = np.mean(data_per_ypix['x'].values)
                yline = yini
                waveline = np.mean(data_per_ypix['wave'].values)
                fluxline = np.mean(data_per_ypix['flux'].values)
                fileorder.write('%f,%f,%f,%f\n' % (float(xline), float(yline), float(waveline), float(fluxline)))
            yini += 1
        omin += 1
        fileorder.close()

    return 0


def get_sampling():
    stardata = pd.read_csv('hcl_spectrum/th_spec_moes.csv', sep=',')
    stardata = stardata.loc[stardata['x'] <= 2048]
    stardata = stardata.loc[stardata['x'] >= 0]
    stardata = stardata.loc[stardata['y'] <= 2048]
    stardata = stardata.loc[stardata['y'] >= 0]
    plt.plot(stardata['x'], stardata['y'], 'k.')
    plt.show()
    orders = np.unique(stardata['order'].values)
    omin = min(orders) + 1
    omax = max(orders)
    fileorder = open('hcl_spectrum/th_spec/spectral_sampling.dat', 'w')
    fileorder.write('order,wave,sampling\n')

    while omin <= omax:
        print(omin)
        data = stardata.loc[stardata['order'].values == omin]
        waves = np.unique(data['wave'].values)
        for k in range(len(waves)):
            wavedata = data.loc[data['wave'] == waves[k]]
            print(waves[k])
            xmin = int(min(wavedata['x'].values))
            xmax = int(max(wavedata['x'].values))

            samp_aux = []
            while xmin < xmax:
                columndata = wavedata.loc[wavedata['x'] < xmin + 1]
                columndata = columndata.loc[columndata['x'] > xmin]
                #print(columndata)
                miny = min(columndata['y'].values)
                maxy = max(columndata['y'].values)
                sampling = maxy - miny
                samp_aux.append(sampling)
                xmin += 1
            samp_aux = np.array(samp_aux)
            sampling = max(samp_aux)
            fileorder.write('%f,%f,%f\n' % (float(omin), float(waves[k]), float(sampling)))
        omin += 1
    fileorder.close()

    return 0


def plotspec():
    basedir = '/home/eduspec/Documentos/moes/fideos_moes/hcl_spectrum/th_spec/'
    omin = 62
    omax = 104
    while omin <= omax:
        data = pd.read_csv(basedir+str(int(omin))+'.dat', sep=',')
        data = data.sort_values(by='wave')
        plt.clf()
        plt.figure(figsize=(12,3))
        plt.plot(data['wave'], data['flux'], 'k-')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Relative intensity')
        plt.tight_layout()
        plt.savefig(basedir+str(int(omin))+'_spec.png')
        plt.clf()
        plt.close()

        plt.clf()
        plt.figure(figsize=(12, 3))
        plt.plot(data['ypix'], data['flux'], 'k-')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Relative intensity')
        plt.tight_layout()
        plt.savefig(basedir + 'pix_spec/' + str(int(omin)) + '_pix.png')
        plt.clf()
        plt.close()

        omin += 1


def gaus(x, x0, sigma):
        return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def samp_measure():
    o = 63
    omin = 63
    omax = 104
    basedir = '/home/eduspec/Documentos/moes/fideos_moes/hcl_spectrum/th_spec/'
    samp_out = []
    while omin <= omax:
        print(omin)
        data = pd.read_csv(basedir + str(int(omin)) + '.dat', sep=',')
        waves = np.unique(data['wave'].values)
        #print(waves)
        for wave in waves:
            #print(waves)
            #w0 = waves[0]
            wavedata = data.loc[data['wave'] == wave]
            wavedata = wavedata.loc[wavedata['flux'] != 0.]
            npix = len(wavedata)
            if npix > 2:
                linearr = []
                maxval = max(wavedata['flux'].values)
                #print(maxval)
                if maxval != 'nan' or float(maxval) != 0.:
                    if maxval == 0.:
                        print(maxval)
                        maxval = 1.
                    wavedata['flux_norm'] = wavedata['flux'].values / maxval
                    for i in range(npix):
                        linearr.append([(min(wavedata['ypix']) - npix + i), 0])

                    for k in range(len(wavedata)):
                        linearr.append([wavedata['ypix'].values[k], wavedata['flux_norm'].values[k]])

                    for j in range(npix):
                        linearr.append([(max(wavedata['ypix']) + j + 1), 0])

                    linearr = np.array(linearr)
                    linearr = pd.DataFrame(linearr, columns=['ypix', 'flux'])
                    linearr.fillna(1)
                    plt.clf()
                    plt.plot(linearr['ypix'], linearr['flux'], '.')
                    # plt.plot(linearr[:, 0], gaus(linearr[:, 0], *popt), 'r-')
                    # plt.show()
                    plt.close()

                    mean = np.mean(linearr['ypix'])
                    sigma = (len(wavedata)) / 3
                    # print(linearr)
                    popt, pcov = curve_fit(gaus, linearr['ypix'], linearr['flux'], p0=[mean, sigma])
                    #popt[1] = popt[1] / 1.5
                    # print(popt)
                    cenpix = popt[0]
                    sigma = popt[1]
                    samp = 2 * np.sqrt(2 * np.log(2)) * sigma
                    # print(omin, wave)
                    samp_out.append(np.array([omin, cenpix, wave, samp]))

        omin += 1

    samp_out = pd.DataFrame(samp_out, columns=['order', 'pix', 'wave', 'sampling'])
    samp_out.to_csv(basedir+'spectral_sampling.csv', index=False)


def sampling_plot():
    basedir = '/home/eduspec/Documentos/moes/fideos_moes/hcl_spectrum/th_spec/'
    sampdata = pd.read_csv(basedir + 'spectral_sampling.dat',sep=',')
    sampdata = sampdata.loc[sampdata['sampling'] > 2.3]
    plt.figure(figsize=[7,4])
    plt.title(r'FIDEOS - f$_{cam}$ = 300 mm')
    plt.plot(sampdata['wave'], sampdata['sampling'],'k.')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Sampling (pix)')
    plt.savefig('/home/eduspec/Documentos/moes/fideos_moes/plots/fideos_samp_f300mm.png')
    plt.show()


def sampling_calc():
    wav_lo = 0.4  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G
    fcol = 762  # mm
    fcam = 230  # mm
    pixsize = 15  # um
    slitsize = 0.1  # mm
    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    avg_samp = []
    while ord_red < ord_blu:
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red)
        wav_max = wav_blz + wav_blz / (2 * ord_red)

        betamin = np.arcsin(ord_red*wav_min*G - np.sin(blaze_angle))
        betamax = np.arcsin(ord_red*wav_max*G - np.sin(blaze_angle))

        Rmin = (np.sin(blaze_angle) + np.sin(betamin))*fcol/(slitsize*np.cos(betamin))
        Rblaze = 2*np.tan(blaze_angle)*fcol/slitsize
        Rmax = (np.sin(blaze_angle) + np.sin(betamax))*fcol/(slitsize*np.cos(betamax))

        linear_disp_min = ord_red * wav_blz * fcam * G / (np.cos(betamin) * Rmin * pixsize * 1e-3)
        linear_disp_cen = ord_red*wav_blz*fcam*G/(np.cos(blaze_angle)*Rblaze*pixsize*1e-3)
        linear_disp_max = ord_red * wav_blz * fcam * G / (np.cos(betamax) * Rmax * pixsize * 1e-3)
        print(linear_disp_min, linear_disp_cen, linear_disp_max)
        avg_samp.append(linear_disp_min)
        avg_samp.append(linear_disp_cen)
        avg_samp.append(linear_disp_max)
        #print(Rmin, Rblaze, Rmax)
        ord_red += 1

    avg_samp = np.array(avg_samp)
    avg_samp = np.average(avg_samp)
    print(avg_samp)


def sampling_moes(fcam):
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
    plt.plot(wave, samp, 'r.', label='Theoretical sampling', alpha = 0.5, zorder = 0)
    plt.plot(wave, sampeout, 'b.', label='Moes sampling', alpha= 0.5, zorder = 1)
    plt.ylim(2.3, 3.)
    plt.ylabel('Spectral sampling')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.savefig()
    plt.show()


if __name__ == '__main__':

    th_moes()
    #sampling_moes(300)
    #spec2D()
    #samp_measure()
    #sampling_plot()
    #sampling_calc()
    #get_sampling()

