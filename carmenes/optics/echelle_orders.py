import numpy as np
import pandas as pd


def init():
    wav_N = 50  # 1575
    wav_lo = 0.55  # in microns
    wav_hi = 1.05
    blaze_angle = 75.76 * np.pi / 180
    G = 31.6 * 1e-3  # lines per um
    d = 1 / G
    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    print('Creating echelle orders...')
    spectrum = []
    order = []
    wave = []
    Hs = []
    DCs = []
    while ord_red < ord_blu + 1:
        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.004
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
    # print(spectrum)
    return slitout


def init2():
    wav_N = 150  # 1575
    wav_lo = 0.36  # in microns
    wav_hi = 0.7
    blaze_angle = 75.76 * np.pi / 180
    G = 31.6 * 1e-3  # lines per um
    d = 1 / G
    ord_blu = int(2 * d * np.sin(blaze_angle) / wav_lo) + 1
    ord_red = int(2 * d * np.sin(blaze_angle) / wav_hi)
    print('Creating echelle orders...')
    spectrum = []
    order = []
    wave = []
    Hs = []
    DCs = []
    while ord_red < ord_blu + 1:

        wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.003
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.0022
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
            spectrum.append(single_element)
            #file.write('%f ' % wav_min)
            wav_min += dwav
            k += 1
        #file.write('\n')
        ord_red += 1

    print('Loading spectrum... Done\n')
    return np.array(spectrum)


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
    blaze_angle = 75.76 * np.pi / 180
    G = 31.6 * 1e-3  # lines per um

    ordout, waveout, x, y, z, dcx, dcy, dcz, fluxout = [], [], [], [], [], [], [], [], []
    print('Creating slit echelle order ', ord_red, 'rv = ', rv)
    wav_blz = 2 * np.sin(blaze_angle) / (G * ord_red)
    wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.006
    wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.006

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


if __name__ == '__main__':

    spec = init()
    print(spec)
    print(min(spec['order']), max(spec['order']))
