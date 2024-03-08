from optics import transform
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import random


def init_line_doppler_full(rv, instr, ord_red):
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0
    diamlo = -0.1
    while diamlo < diam:
        slitgrid.append([diamlo, 0.])
        diamlo += 0.01
        
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    # outpath = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # outpath = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    wav_N = 5000  # 1575
    wav_lo = 0.36  # in microns
    wav_hi = 0.71
    blaze_angle = 76. * np.pi / 180
    G = 41.6 * 1e-3  # lines per um

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    #print('Creating slit echelle order ', ord_red, 'rv = ', rv)

    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002
    # print(wav_blz, wav_min, wav_max)
    dwav = (wav_max - wav_min) / wav_N
    k = 0

    while k <= wav_N:
        for l in range(len(slitgrid)):
            ordout.append(ord_red)
            waveout.append(wav_min)
            x.append(slitgrid[l][0])
            y.append(slitgrid[l][1])
            z.append(0)
            dcx.append(0)
            dcy.append(0)
            dcz.append(1.)
            fluxout.append(1.)
            # file.write('%f ' % wav_min)

        wav_min += dwav
        k += 1
        # file.write('\n')

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dcx
    slitout['dy'] = dcy
    slitout['dz'] = dcz
    slitout['flux'] = fluxout

    #print('Output array created...')
    return slitout


def init_points_doppler_full(rv, instr, ord_red):
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0

    slitgrid.append([0., 0.05])
    slitgrid.append([0., -0.05])
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    # outpath = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # outpath = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    wav_N = 50  # 1575
    wav_lo = 0.36  # in microns
    wav_hi = 0.71
    blaze_angle = 76. * np.pi / 180
    G = 41.6 * 1e-3  # lines per um

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    #print('Creating slit echelle order ', ord_red, 'rv = ', rv)

    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002
    # print(wav_blz, wav_min, wav_max)
    dwav = (wav_max - wav_min) / wav_N
    k = 0

    while k <= wav_N:
        for l in range(len(slitgrid)):
            ordout.append(ord_red)
            waveout.append(wav_min)
            x.append(slitgrid[l][0])
            y.append(slitgrid[l][1])
            z.append(0)
            dcx.append(0)
            dcy.append(0)
            dcz.append(1.)
            fluxout.append(1.)
            # file.write('%f ' % wav_min)

        wav_min += dwav
        k += 1
        # file.write('\n')

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dcx
    slitout['dy'] = dcy
    slitout['dz'] = dcz
    slitout['flux'] = fluxout

    #print('Output array created...')
    return slitout


def init_points_doppler_sampling(rv, instr, ord_red):
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0

    slitgrid.append([0., 0.05])
    slitgrid.append([0., -0.05])
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    # outpath = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # outpath = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    wav_N = 50  # 1575
    wav_lo = 0.36  # in microns
    wav_hi = 0.71
    blaze_angle = 76. * np.pi / 180
    G = 41.6 * 1e-3  # lines per um

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    #print('Creating slit echelle order ', ord_red, 'rv = ', rv)

    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002
    # print(wav_blz, wav_min, wav_max)
    dwav = (wav_max - wav_min) / wav_N
    k = 0
    randwaves = [random.uniform(wav_min, wav_max) for _ in range(wav_N)]
    #print(randwaves)
    for wave in randwaves:
        for l in range(len(slitgrid)):
            ordout.append(ord_red)
            waveout.append(wave)
            x.append(slitgrid[l][0])
            y.append(slitgrid[l][1])
            z.append(0)
            dcx.append(0)
            dcy.append(0)
            dcz.append(1.)
            fluxout.append(1.)

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dcx
    slitout['dy'] = dcy
    slitout['dz'] = dcz
    slitout['flux'] = fluxout

    #print('Output array created...')
    return slitout


def init():
    wav_N = 1200  # 1575
    wav_lo = 0.36  # in microns
    wav_hi = 0.71
    blaze_angle = 76. * np.pi / 180
    G = 41.6 * 1e-3  # lines per um
    d = 1 / G
    # n=

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo)

    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    # print('Creating echelle orders...')
    spectrum = []
    order = []
    wave = []
    Hs = []
    DCs = []
    x = []
    y = []
    z = []
    dx = []
    dy = []
    dz = []
    flux = []
    while ord_red < ord_blu + 1:

        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.008
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.008
        dwav = (wav_max - wav_min) / wav_N
        k = 0

        while k <= wav_N:
            H = np.zeros([3])
            DC = np.zeros([3])
            order.append(ord_red)
            wave.append(wav_min)
            Hs.append(H)
            DCs.append(DC)
            single_element = (ord_red, wav_min)
            x.append(0)
            y.append(0)
            z.append(0)
            dx.append(0)
            dy.append(0)
            dz.append(1.)
            flux.append(1.)
            spectrum.append(single_element)
            # file.write('%f ' % wav_min)
            wav_min += dwav
            k += 1
        # file.write('\n')
        ord_red += 1

    #print('Loading spectrum... Done\n')

    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = x
    specout['y'] = y
    specout['z'] = z
    specout['dx'] = dx
    specout['dy'] = dy
    specout['dz'] = dz
    specout['flux'] = flux

    return specout



def init_g2mask(mask, ord):

    # print('Creating echelle orders...')
    spectrum = []
    order = []
    wave = []
    Hs = []
    DCs = []
    x = []
    y = []
    z = []
    dx = []
    dy = []
    dz = []
    flux = []

    for k in range(len(mask)):
        H = np.zeros([3])
        DC = np.zeros([3])
        order.append(ord)
        wave.append(mask['wave'].values[k])
        Hs.append(H)
        DCs.append(DC)
        single_element = (ord, mask['wave'].values[k])
        x.append(0)
        y.append(0)
        z.append(0)
        dx.append(0)
        dy.append(0)
        dz.append(1.)
        flux.append(1.)
        spectrum.append(single_element)

    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = x
    specout['y'] = y
    specout['z'] = z
    specout['dx'] = dx
    specout['dy'] = dy
    specout['dz'] = dz
    specout['flux'] = flux

    return specout



def dispersion(s, params, dpix):
    alpha = np.full(len(s), np.abs(params[7] * np.pi / 180))
    g = np.full(len(s), params[5] * 1e-3)
    gamma = np.full(len(s), params[6] * np.pi / 180)
    f = np.full(len(s), 455.)
    pix_size = np.full(len(s), 15., dtype=float)
    beta = np.arcsin(np.abs(s[:, 0]) * g * s[:, 1] / np.cos(gamma) - np.sin(alpha))
    dg = s[:, 0] * g * f * np.full(len(s), 1e3) / (np.cos(gamma) * np.cos(beta) * pix_size)
    dl = dpix / dg
    c = np.full(len(s), 3e8)
    dv = c * dl / s[:, 1]
    return dv


def init_cone(frat):
    # Creation of angles grid depending on focal ratio
    na = 1 / (2 * frat)
    theta = np.arcsin(na)
    dtheta = 0.1
    tx_max = theta * 180 / np.pi
    ty_max = theta * 180 / np.pi
    tx_min = -theta * 180 / np.pi
    ty_min = -theta * 180 / np.pi
    Tgrid = []
    while tx_min < tx_max:
        # print(tx_min)
        while ty_min < ty_max:
            Taux = np.array([tx_min * np.pi / 180, ty_min * np.pi / 180, 0.])
            # print(Taux)
            Tgrid.append(Taux)
            ty_min += dtheta
        tx_min += dtheta
        ty_min = -theta * 180 / np.pi

    Tgrid = np.array(Tgrid)

    # We create echelle orders
    wav_N = 3  # 1575
    wav_lo = 0.4  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    spectrum = []
    while ord_red < ord_blu + 1:

        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.003
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.0022
        dwav = (wav_max - wav_min) / wav_N
        k = 0
        while k <= wav_N:
            for l in range(len(Tgrid)):
                H = np.zeros([3])
                DC = np.zeros([3])
                DC[2] = -1.
                # print(Tgrid[l])
                DCt = transform.transform_single(DC, Tgrid[l])
                # int(DCt)
                # print(DCt[0])
                single_element = (ord_red, wav_min, H[0], H[1], H[2], DCt[0][0], DCt[0][1], DCt[0][2])
                spectrum.append(np.array(single_element))
            wav_min += dwav
            k += 1
        # file.write('\n')
        ord_red += 1

    # print(Tgrid)
    spectrum = np.array(spectrum)
    spectrum = pd.DataFrame(spectrum, columns=['order', 'wave', 'x', 'y', 'z', 'dx', 'dy', 'dz'])
    return spectrum


def init_slit():
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0
    thetamax = 360.
    while daux <= diam:
        thetamin = 0.
        while thetamin <= thetamax:
            if thetamin <= 90:
                x = daux * np.cos(thetamin * np.pi / 180) - diam
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2
                slitgrid.append(np.array([x, y]))
            elif 90 < thetamin <= 180:
                x = daux * np.cos(thetamin * np.pi / 180) - diam
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2
                slitgrid.append(np.array([x, y]))
                if thetamin == 180.:
                    x = daux * np.cos(thetamin * np.pi / 180) + diam
                    y = daux * np.sin(thetamin * np.pi / 180) + diam / 2
                    slitgrid.append(np.array([x, y]))
            elif 180 < thetamin <= 270:
                x = daux * np.cos(thetamin * np.pi / 180) + diam
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2
                slitgrid.append(np.array([x, y]))
            elif 270 < thetamin <= 360:
                x = daux * np.cos(thetamin * np.pi / 180) + diam
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2
                slitgrid.append(np.array([x, y]))
            thetamin += 15.
        daux += 0.025
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()

    wav_N = 1600  # 1575
    wav_lo = 0.38  # in microns
    wav_hi = 0.68
    blaze_angle = 76. * np.pi / 180
    G = 41.6 * 1e-3  # lines per um
    d = 1 / G

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo)
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    spectrum = []

    while ord_red < ord_blu + 1:
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002
        dwav = (wav_max - wav_min) / wav_N
        k = 0
        while k <= wav_N:
            for l in range(len(slitgrid)):
                H = np.zeros([3])
                H[0] = slitgrid[l][0]
                H[1] = slitgrid[l][1]
                DC = np.zeros([3])
                DC[2] = 1.
                single_element = (ord_red, wav_min, H[0], H[1], H[2], DC[0], DC[1], DC[2], 1.)
                spectrum.append(np.array(single_element))
            wav_min += dwav
            k += 1
        # file.write('\n')
        ord_red += 1

    # print(Tgrid)
    spectrum = np.array(spectrum)
    spectrum = pd.DataFrame(spectrum, columns=['order', 'wave', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'flux'])

    return spectrum


def init_stellar():
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
                x = daux * np.cos(thetamin * np.pi / 180) - diam
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2
                slitgrid.append(np.array([x, y]))
            elif 90 <= thetamin < 180:
                x = daux * np.cos(thetamin * np.pi / 180) - diam
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2
                slitgrid.append(np.array([x, y]))
            elif 180 <= thetamin < 270:
                x = daux * np.cos(thetamin * np.pi / 180) + diam
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2
                slitgrid.append(np.array([x, y]))
            else:
                x = daux * np.cos(thetamin * np.pi / 180) + diam
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2
                slitgrid.append(np.array([x, y]))

            thetamin += 36.
        thetamin = 0.
        daux += 0.025
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()
    basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    stellarpath = '/home/eduspec/Documentos/moes/fideos_moes/stellar_spectrum/'
    stellar_spec = pd.read_csv(stellarpath + 'stellar_spectrum_norm.dat', sep=',')
    #stellarslit_out = open(stellarpath + 'stellar_slit_spec_rv_500ms.dat', 'w')
    rv = 500.0  # m/s
    #stellarslit_out.write('order,wave,x,y,z,dx,dy,dz,flux\n')
    wav_lo = 0.38  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    # spectrum = []

    while ord_red < ord_blu + 1:
        print(ord_red)
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.003
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.0022

        wmin = wav_min * 1e4
        wmax = wav_max * 1e4
        # print(wmin, wmax)
        stellardata = stellar_spec.loc[stellar_spec['wave'] < wmax]
        stellardata = stellardata.loc[stellardata['wave'] > wmin]
        stellardata['wave_new'] = stellardata['wave'] * (1 + rv / 3.e8)
        ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
        for k in range(len(stellardata)):
            for l in range(len(slitgrid)):
                H = np.zeros([3])
                H[0] = slitgrid[l][0]
                H[1] = slitgrid[l][1]
                DC = np.zeros([3])
                DC[2] = -1.
                # single_element = (ord_red, stellardata['wave'].values[k], H[0], H[1], H[2], DC[0], DC[1], DC[2])
                ordout.append(float(ord_red))
                waveout.append(float((stellardata['wave_new'].values[k]) * 1e-4))
                x.append(float(H[0]))
                y.append(float(H[1]))
                z.append(float(H[2]))
                dcx.append(float(DC[0]))
                dcy.append(float(DC[1]))
                dcz.append(float(DC[2]))
                fluxout.append(float(stellardata['flux_norm'].values[k]))
                # stellarslit_out.write('%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (float(ord_red), float((stellardata['wave_new'].values[k]) * 1e-4), float(H[0]), float(H[1]), float(H[2]), float(DC[0]), float(DC[1]), float(DC[2]), float(stellardata['flux_norm'].values[k])))
                # spectrum.append(np.array(single_element))

        ord_red += 1

    return 0


def init_stellar_doppler_simple(rv, instr, ord_red):
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    # outpath = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # outpath = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    # stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    stardir = 'stellar_template/'

    stellar_spec = pd.read_csv(stardir + 'stellar_template_resampled.tsv',
                               sep=',')

    # if not os.path.isdir(outpath):
    #    os.mkdir(outpath)
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    print('Creating slit echelle order ', ord_red, 'rv = ', rv)
    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002

    wmin = wav_min * 1e4
    wmax = wav_max * 1e4
    stellardata = stellar_spec.loc[stellar_spec['wave'] < wmax]
    stellardata = stellardata.loc[stellardata['wave'] > wmin]
    stellardata['wave_new'] = stellardata['wave'] * (1 + rvfinal / 3.e8)
    for k in range(len(stellardata)):
        H = np.zeros([3])
        H[0] = 0
        H[1] = 0
        H[2] = 0
        DC = np.zeros([3])
        if instr == 'platospec':
            DC[2] = 1.
        elif instr == 'fideos':
            DC[2] = -1.

        ordout.append(float(ord_red))
        waveout.append(float((stellardata['wave_new'].values[k]) * 1e-4))
        x.append(float(H[0]))
        y.append(float(H[1]))
        z.append(float(H[2]))
        dcx.append(float(DC[0]))
        dcy.append(float(DC[1]))
        dcz.append(float(DC[2]))
        fluxout.append(float(stellardata['flux'].values[k]))

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dcx
    slitout['dy'] = dcy
    slitout['dz'] = dcz
    slitout['flux'] = fluxout

    print('Slit array created...')
    return slitout


def init_stellar_doppler_full(rv, instr, ord_red):
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0
    thetamax = 360.
    while daux <= diam:
        thetamin = 0.
        while thetamin <= thetamax:
            if thetamin <= 90:
                x = daux * np.cos(thetamin * np.pi / 180) - diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            elif 90 < thetamin <= 180:
                x = daux * np.cos(thetamin * np.pi / 180) - diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2 + decy
                slitgrid.append(np.array([x, y]))
                if thetamin == 180.:
                    x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                    y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                    slitgrid.append(np.array([x, y]))
            elif 180 < thetamin <= 270:
                x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            elif 270 < thetamin <= 360:
                x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            thetamin += 10.
        daux += 0.015
    slitgrid = np.array(slitgrid)
    #plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    #plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    #outpath = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    #outpath = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    #stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    stardir = 'stellar_template/'

    stellar_spec = pd.read_csv(stardir + 'stellar_template_V3.tsv',
                               sep=',')

    #if not os.path.isdir(outpath):
    #    os.mkdir(outpath)
    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    print('Creating slit echelle order ', ord_red, 'rv = ', rv)
    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002

    wmin = wav_min * 1e4
    wmax = wav_max * 1e4
    stellardata = stellar_spec.loc[stellar_spec['wave'] < wmax]
    stellardata = stellardata.loc[stellardata['wave'] > wmin]
    stellardata['wave_new'] = stellardata['wave'] * (1 + rvfinal / 3.e8)
    for k in range(len(stellardata)):
        for l in range(len(slitgrid)):
            H = np.zeros([3])
            H[0] = slitgrid[l][0]
            H[1] = slitgrid[l][1]
            DC = np.zeros([3])
            if instr == 'platospec':
                DC[2] = 1.
            elif instr == 'fideos':
                DC[2] = -1.

            ordout.append(float(ord_red))
            waveout.append(float((stellardata['wave_new'].values[k]) * 1e-4))
            x.append(float(H[0]))
            y.append(float(H[1]))
            z.append(float(H[2]))
            dcx.append(float(DC[0]))
            dcy.append(float(DC[1]))
            dcz.append(float(DC[2]))
            fluxout.append(float(stellardata['flux'].values[k]))

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dcx
    slitout['dy'] = dcy
    slitout['dz'] = dcz
    slitout['flux'] = fluxout

    print('Output array created...')
    return slitout


def init_stellar_doppler_order(rv, fcam, omin):
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

            thetamin += 36.
        thetamin = 0.
        daux += 0.025
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    outpath = '/media/eduspec/TOSHIBA EXT/fideos_moes/data/f' + str(int(fcam)) + 'mm/slit_files/' + str(int(rv)) + '/'
    # outpath = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/slit_files/'+str(int(rv))+'/'
    # outpath = '/home/eduspec/Documentos/moes/fideos_moes/stellar_spectrum/f230mm/rv_specs/'
    stellarpath = '/home/eduspec/Documentos/moes/fideos_moes/stellar_spectrum/'

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    stellar_spec = pd.read_csv(stellarpath + 'stellar_spectrum_norm.dat', sep=',')

    wav_lo = 0.38  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    # spectrum = []
    ord_red = omin
    ord_blu = omin
    while ord_red <= ord_blu:
        print('Creating slit echelle order ', ord_red, ', fcam = ', fcam, ', rv = ', rv)
        stellarslit_out = open(outpath + str(int(ord_red)) + '_slit.tsv', 'w')
        stellarslit_out.write('order,wave,x,y,z,dx,dy,dz,flux\n')
        print('Output file created...')
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.0018
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.0018

        wmin = wav_min * 1e4
        wmax = wav_max * 1e4
        # print(wmin, wmax)
        stellardata = stellar_spec.loc[stellar_spec['wave'] < wmax]
        stellardata = stellardata.loc[stellardata['wave'] > wmin]
        stellardata['wave_new'] = stellardata['wave'] * (1 + rvfinal / 3.e8)
        for k in range(len(stellardata)):
            for l in range(len(slitgrid)):
                H = np.zeros([3])
                H[0] = slitgrid[l][0]
                H[1] = slitgrid[l][1]
                DC = np.zeros([3])
                DC[2] = -1.
                # single_element = (ord_red, stellardata['wave'].values[k], H[0], H[1], H[2], DC[0], DC[1], DC[2])
                stellarslit_out.write('%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
                    float(ord_red), float((stellardata['wave_new'].values[k]) * 1e-4), float(H[0]), float(H[1]),
                    float(H[2]), float(DC[0]), float(DC[1]), float(DC[2]), float(stellardata['flux_norm'].values[k])))
                # spectrum.append(np.array(single_element))
        stellarslit_out.close()
        ord_red += 1

    return 0


def init_stellar_doppler_single(rv, omin, detno):
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    #outpath = 'data/slit_files/' + str(int(rv)) + '/'
    # outpath = '/home/eduspec/Documentos/moes/fideos_moes/data/f' + str(int(fcam)) + 'mm/slit_files/'+str(int(rv))+'/'
    # outpath = '/home/eduspec/Documentos/moes/fideos_moes/stellar_spectrum/f230mm/rv_specs/'
    stellarpath = 'stellar_template/'

    #if not os.path.isdir(outpath):
    #    os.mkdir(outpath)

    stellar_spec = pd.read_csv(stellarpath + 'stellar_template.tsv', sep=',')

    wav_lo = 0.38  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G

    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    # spectrum = []
    ord_red = omin
    ord_blu = omin
    x, y, z, dx, dy, dz, ordout, waveout, fluxout = [], [], [], [], [], [], [], [], []
    while ord_red <= ord_blu:
        print('Creating slit echelle order ', ord_red, ', rv = ', rv)

        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.0018
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.0018

        wmin = wav_min * 1e4
        wmax = wav_max * 1e4
        # print(wmin, wmax)
        stellardata = stellar_spec.loc[stellar_spec['wave'] < wmax]
        stellardata = stellardata.loc[stellardata['wave'] > wmin]
        stellardata['wave_new'] = stellardata['wave'] * (1 + rvfinal / 3.e8)
        for k in range(len(stellardata)):
            H = np.zeros([3])
            H[0] = 0
            H[1] = 0
            DC = np.zeros([3])
            DC[2] = -1.
            # single_element = (ord_red, stellardata['wave'].values[k], H[0], H[1], H[2], DC[0], DC[1], DC[2])
            x.append(H[0])
            y.append(H[1])
            z.append(H[2])
            dx.append(DC[0])
            dy.append(DC[1])
            dz.append(DC[2])
            ordout.append(ord_red)
            waveout.append(stellardata['wave_new'].values[k])
            fluxout.append(stellardata['flux'].values[k])
        ord_red += 1

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dx
    slitout['dy'] = dy
    slitout['dz'] = dz
    slitout['flux'] = fluxout

    return slitout


def init_thar_doppler_full(rv, instr, ord_red):
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0
    thetamax = 360.
    while daux <= diam:
        thetamin = 0.
        while thetamin <= thetamax:
            if thetamin <= 90:
                x = daux * np.cos(thetamin * np.pi / 180) - diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            elif 90 < thetamin <= 180:
                x = daux * np.cos(thetamin * np.pi / 180) - diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) - diam / 2 + decy
                slitgrid.append(np.array([x, y]))
                if thetamin == 180.:
                    x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                    y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                    slitgrid.append(np.array([x, y]))
            elif 180 < thetamin <= 270:
                x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            elif 270 < thetamin <= 360:
                x = daux * np.cos(thetamin * np.pi / 180) + diam + decx
                y = daux * np.sin(thetamin * np.pi / 180) + diam / 2 + decy
                slitgrid.append(np.array([x, y]))
            thetamin += 10.
        daux += 0.015
    slitgrid = np.array(slitgrid)
    #plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    #plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
    # basepath = '/home/eduspec/Documentos/moes/fideos_moes/'
    #outpath = '/media/eduspec/TOSHIBA EXT/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    #outpath = '/home/eduspec/Documentos/moes/platospec/data/f' + str(int(fcam)) + 'mm/slit/' + str(int(rv)) + '/'
    #stardir = '/home/eduspec/Documentos/moes/P_s5700g4.50z0.0t0.97_a0.00c0.00n0.00o0.00_VIS.spec.flat/'
    stardir = 'stellar_template/'

    thar_spec = pd.read_csv(stardir + 'thar_list.txt',
                               delim_whitespace=True, names=['wave', 'line'])
    #thar_spec['wave'] = thar_spec['wave'] * 1e-4

    if instr == 'fideos':
        blaze_angle = 70. * np.pi / 180
        G = 44.41 * 1e-3  # lines per um
        d = 1 / G
        fcol = 762

    elif instr == 'platospec':
        blaze_angle = 76. * np.pi / 180
        G = 41.6 * 1e-3  # lines per um
        d = 1 / G
        fcol = 876

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    print('Creating slit echelle order ', ord_red, 'rv = ', rv)
    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.002
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.002

    wmin = wav_min * 1e4
    wmax = wav_max * 1e4
    thardata = thar_spec.loc[thar_spec['wave'] < wmax]
    thardata = thardata.loc[thardata['wave'] > wmin]
    
    for k in range(len(thardata)):
        for l in range(len(slitgrid)):
            H = np.zeros([3])
            H[0] = slitgrid[l][0]
            H[1] = slitgrid[l][1]
            DC = np.zeros([3])
            if instr == 'platospec':
                DC[2] = 1.
            elif instr == 'fideos':
                DC[2] = -1.

            ordout.append(float(ord_red))
            waveout.append(float((thardata['wave'].values[k]) * 1e-4))
            x.append(float(H[0]))
            y.append(float(H[1]))
            z.append(float(H[2]))
            dcx.append(float(DC[0]))
            dcy.append(float(DC[1]))
            dcz.append(float(DC[2]))
            fluxout.append(1.)

    slitout = pd.DataFrame()
    slitout['order'] = ordout
    slitout['wave'] = waveout
    slitout['x'] = x
    slitout['y'] = y
    slitout['z'] = z
    slitout['dx'] = dcx
    slitout['dy'] = dcy
    slitout['dz'] = dcz
    slitout['flux'] = fluxout

    print('Output array created...')
    return slitout


if __name__ == '__main__':
    # init_stellar()
    slit = init_thar_doppler_full(0, 'platospec', 71)
    print(slit)
