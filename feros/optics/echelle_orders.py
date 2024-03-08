import numpy as np
import math as m
import pandas as pd
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
import matplotlib
matplotlib.rc('font', **font)


def init_points_doppler_full(rv, instr, ord_red):
    slitgrid = []
    diam = 0.1
    decx = 0.0
    decy = 0.0
    daux = 0.0

    slitgrid.append([0., 0.06])
    slitgrid.append([0., -0.06])
    slitgrid = np.array(slitgrid)
    # plt.plot(slitgrid[:,0], slitgrid[:,1],'.')
    # plt.show()
    rvbase = 0.0  # m/s
    rvfinal = rvbase + rv
   
    wav_lo = 0.36  # in microns
    wav_hi = 0.71
    blaze_angle = 63.4 * np.pi / 180
    G = 79 * 1e-3  # lines per um
    d = 1 / G

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    #print('Creating slit echelle order ', ord_red, 'rv = ', rv)
    wav_N = 100
    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.003
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.003
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
            dcz.append(-1.)
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


def init():
    wav_N = 500  # 1575
    wav_lo = 0.36 # in microns
    wav_hi = 0.7
    blaze_angle = 63.4 * np.pi / 180
    G = 79 * 1e-3  # lines per um
    d = 1 / G
    ord_blu = int(2*d*np.sin(blaze_angle)/wav_lo) + 1
    ord_red = int(2*d*np.sin(blaze_angle)/wav_hi)
    print('Creating echelle orders...')
    spectrum = []
    order = []
    wave = []
    Hs = []
    DCs = []
    while ord_red < ord_blu + 1:
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.005
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.004
        dwav = (wav_max - wav_min) / wav_N
        k = 0
        while k <= wav_N:
            H = np.zeros([3])
            DC = np.zeros([3])
            order.append(ord_red)
            wave.append(float(f'{float(wav_min):.5f}'))
            Hs.append(H)
            DCs.append(DC)
            single_element = (ord_red, wav_min)
            spectrum.append(single_element)
            wav_min += dwav
            k += 1
        ord_red += 1


    slitout = pd.DataFrame()
    slitout['order'] = order
    slitout['wave'] = wave
    slitout['x'] = np.full(len(order), 0.)
    slitout['y'] = np.full(len(order), 0.)
    slitout['z'] = np.full(len(order), 0.)
    slitout['dx'] = np.full(len(order), 0.)
    slitout['dy'] = np.full(len(order), 0.)
    slitout['dz'] = np.full(len(order), 1.)
    slitout['flux'] = np.full(len(order), 1.)
    print('Loading spectrum... Done\n')
    #print(spectrum)
    return slitout


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


def init_stellar_doppler_simple(rv, ord_red):
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
    blaze_angle = 63.4 * np.pi / 180
    G = 79. * 1e-3  # lines per um

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    print('Creating slit echelle order ', ord_red, 'rv = ', rv)
    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.005
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.004

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



if __name__ == '__main__':
    spectrum = init()
