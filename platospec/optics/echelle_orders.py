import numpy as np
from optics import transform
import pandas as pd


def init():
    file = open('echelle_orders_vis.txt', 'w')
    file.write('order wave_min wave_cen wave_max\n')
    file_0 = open('echelle_blaze_wave.txt', 'w')
    wav_N = 3  # 1575
    wav_lo = 0.4  # in microns
    wav_hi = 0.68
    blaze_angle = 70. * np.pi / 180
    G = 44.41 * 1e-3  # lines per um
    d = 1 / G
    # n=

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
        wav_min = wav_blz - wav_blz / (2 * ord_red) - 0.003
        wav_max = wav_blz + wav_blz / (2 * ord_red) + 0.0022
        dwav = (wav_max - wav_min) / wav_N
        file.write('%d     %.5f  %.5f  %.5f\n' % (ord_red, wav_min, wav_blz, wav_max))
        # file_0.write('%d %f %f %f\n' %(ord_red, wav_min ,wav_blz, wav_max))
        file_0.write('%d %f\n' % (ord_red, wav_blz))
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

    file.close()
    file_0.close()
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


def init_cone(frat):

    # Creation of angles grid depending on focal ratio
    na = 1/(2*frat)
    theta = np.arcsin(na)
    dtheta = 0.1
    tx_max = theta*180/np.pi
    ty_max = theta*180/np.pi
    tx_min = -theta*180/np.pi
    ty_min = -theta*180/np.pi
    Tgrid = []
    while tx_min < tx_max:
        #print(tx_min)
        while ty_min < ty_max:
            Taux = np.array([tx_min*np.pi/180, ty_min*np.pi/180, 0.])
            #print(Taux)
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
                #print(Tgrid[l])
                DCt = transform.transform_single(DC, Tgrid[l])
                #int(DCt)
                #print(DCt[0])
                single_element = (ord_red, wav_min, H[0], H[1], H[2], DCt[0][0], DCt[0][1], DCt[0][2])
                spectrum.append(np.array(single_element))
            wav_min += dwav
            k += 1
        #file.write('\n')
        ord_red += 1

    #print(Tgrid)
    spectrum = np.array(spectrum)
    spectrum = pd.DataFrame(spectrum, columns=['order','wave','x','y','z','dx','dy','dz'])
    print(spectrum)

    return spectrum

if __name__ == '__main__':

    spec = init_cone(22)

