#import pyzmx
from optics import collimator
from optics import trace
import numpy as np
import pandas as pd
import echelle_orders
from optics import transform
import matplotlib.pyplot as plt
from optics import echelle
from optics import prisms
from optics import paraxial
from decimal import *
from optics import CCD
getcontext().prec = 20


def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})


def tracing(spectrum):  # (params, fib, temps):
    #
    # Variables initialization
    #
    # Variable definition, spectrum data extraction
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    order[:] = spectrum[:, 0]
    wave[:] = spectrum[:, 1]

    # Slit initialization
    H_init[:, 0] = np.zeros(len(spectrum))
    H_init[:, 1] = np.zeros(len(spectrum))
    H_init[:, 2] = np.zeros(len(spectrum))

    DC_init[:, 0] = np.zeros(len(spectrum))
    DC_init[:, 1] = np.zeros(len(spectrum))
    DC_init[:, 2] = np.zeros(len(spectrum))
    DC_init[:,2] = np.full(len(spectrum), -1)

    T_slit = np.asarray([0.*np.pi/180, 4.*np.pi/180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 762.
    coll_tilt_x = 0
    coll_tilt_y = 0
    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H2 = trace.to_next_surface(H_init, DC_slit, d_slit_col)
    rad = -762*2
    H3, DC3 = collimator.DCcoll(H2, DC_slit, T_coll, rad)

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -800)
    z_ech = d_col_ech + H3[:, 2]
    #print(z_ech)
    G = 44.41*1e-3  # lines/um
    #G = 1 / d
    H4 = trace.to_next_surface(H3, DC3, z_ech)
    ech_blaze = 70.  # deg
    ech_gamma = 4.  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H4, DC4 = echelle.diffraction(H4, DC3, T_echelle, order, wave, G)

    # todo bien hasta aca
    # tracing to cross-dispersing prisms
    d_ech_prm = np.full(len(H_init), 300)
    H5 = trace.to_next_surface(H4, DC4, d_ech_prm)
    #print(DC4)
    prisms_material = 'SF11'
    dec_x = 95.
    dec_y = 0.
    apex_angle = -34.6
    Tin = np.array([0.0*np.pi/180, 8.*np.pi/180, 0.0*np.pi/180])
    H5, DC5 = prisms.tracing(H5, DC4, Tin, wave, prisms_material, apex_angle, dec_x, dec_y)
    H5 = np.array(H5)
    H6, DC6 = paraxial.tracing(H5, DC5, 300, 300)
    #print(H5, DC5)
    Hout = H6.copy()

    return Hout


def tracing_full(spectrum):  # (params, fib, temps):
    #
    # Variables initialization
    #
    set_numpy_decimal_places(20, 0)
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    flux = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values
    flux[:] = spectrum['flux'].values

    # Slit initialization
    H_init[:, 0] = spectrum['x'].values
    H_init[:, 1] = spectrum['y'].values
    H_init[:, 2] = spectrum['z'].values

    DC_init[:, 0] = spectrum['dx'].values
    DC_init[:, 1] = spectrum['dy'].values
    DC_init[:, 2] = spectrum['dz'].values
    T_slit = np.asarray([0.*np.pi/180, 4.*np.pi/180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 762.  # 1590
    coll_tilt_x = 0
    coll_tilt_y = 0
    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H2 = trace.to_next_surface(H_init, DC_slit, d_slit_col)
    rad = -762*2
    H3, DC3 = collimator.DCcoll(H2, DC_slit, T_coll, rad)

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -800)
    z_ech = d_col_ech + H3[:, 2]
    #print(z_ech)
    G = 44.41*1e-3  # lines/um
    #G = 1 / d
    H4 = trace.to_next_surface(H3, DC3, z_ech)

    ech_blaze = 70.  # deg
    ech_gamma = 4.  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H4, DC4 = echelle.diffraction(H4, DC3, T_echelle, order, wave, G)

    # todo bien hasta aca
    # tracing to cross-dispersing prisms
    d_ech_prm = np.full(len(H_init), 300)
    H5 = trace.to_next_surface(H4, DC4, d_ech_prm)
    #print(DC4)
    prisms_material = 'SF11'
    dec_x = 95.
    dec_y = 0.
    apex_angle = -34.6
    Tin = np.array([0.0*np.pi/180, 8.*np.pi/180, 0.0*np.pi/180])
    H5, DC5 = prisms.tracing(H5, DC4, Tin, wave, prisms_material, apex_angle, dec_x, dec_y)
    H5 = np.array(H5)
    H6, DC6 = paraxial.tracing(H5, DC5, 300., 300.)
    #print(H6)
    #print(H5, DC5)
    #Hout = H6.copy()
    #DCout = DC6.copy()
    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = H6[:, 0]
    specout['y'] = H6[:, 1]
    specout['z'] = H6[:, 2]
    specout['dx'] = DC6[:, 0]
    specout['dy'] = DC6[:, 1]
    specout['dz'] = DC6[:, 2]
    specout['flux'] = flux
    specout = CCD.mm2pix(specout)
    
    return specout


def tracing_full_fcam(spectrum, fcam):  # (params, fib, temps):
    #
    # Variables initialization
    #
    set_numpy_decimal_places(20, 0)
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    flux = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values
    flux[:] = spectrum['flux'].values

    # Slit initialization
    H_init[:, 0] = spectrum['x'].values
    H_init[:, 1] = spectrum['y'].values
    H_init[:, 2] = spectrum['z'].values

    DC_init[:, 0] = spectrum['dx'].values
    DC_init[:, 1] = spectrum['dy'].values
    DC_init[:, 2] = spectrum['dz'].values
    T_slit = np.asarray([0. * np.pi / 180, 4. * np.pi / 180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 762.  # 1590
    coll_tilt_x = 0
    coll_tilt_y = 0
    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H2 = trace.to_next_surface(H_init, DC_slit, d_slit_col)
    rad = -762 * 2
    H3, DC3 = collimator.DCcoll(H2, DC_slit, T_coll, rad)

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -800)
    z_ech = d_col_ech + H3[:, 2]
    # print(z_ech)
    G = 44.41 * 1e-3  # lines/um
    # G = 1 / d
    H4 = trace.to_next_surface(H3, DC3, z_ech)

    ech_blaze = 70.  # deg
    ech_gamma = 4.  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H4, DC4 = echelle.diffraction(H4, DC3, T_echelle, order, wave, G)

    # todo bien hasta aca
    # tracing to cross-dispersing prisms
    d_ech_prm = np.full(len(H_init), 300)
    H5 = trace.to_next_surface(H4, DC4, d_ech_prm)
    # print(DC4)
    prisms_material = 'SF11'
    dec_x = 95.
    dec_y = 0.
    apex_angle = -34.6
    Tin = np.array([0.0 * np.pi / 180, 8. * np.pi / 180, 0.0 * np.pi / 180])
    H5, DC5 = prisms.tracing(H5, DC4, Tin, wave, prisms_material, apex_angle, dec_x, dec_y)
    H5 = np.array(H5)
    H6, DC6 = paraxial.tracing(H5, DC5, fcam, fcam)
    # print(H6)
    # print(H5, DC5)
    # Hout = H6.copy()
    # DCout = DC6.copy()
    ccd_decx = 0.4
    ccd_decy = -5.
    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = H6[:, 0] + ccd_decx
    specout['y'] = H6[:, 1] + ccd_decy
    specout['z'] = H6[:, 2]
    specout['dx'] = DC6[:, 0]
    specout['dy'] = DC6[:, 1]
    specout['dz'] = DC6[:, 2]
    specout['flux'] = flux
    specout = CCD.mm2pix(specout)

    return specout


def raytrace(spectrum, det):  # (params, fib, temps):
    #
    # Variables initialization
    #
    set_numpy_decimal_places(20, 0)
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    flux = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values
    flux[:] = spectrum['flux'].values

    # Slit initialization
    H_init[:, 0] = spectrum['x'].values
    H_init[:, 1] = spectrum['y'].values
    H_init[:, 2] = spectrum['z'].values

    DC_init[:, 0] = spectrum['dx'].values
    DC_init[:, 1] = spectrum['dy'].values
    DC_init[:, 2] = spectrum['dz'].values
    T_slit = np.asarray([0. * np.pi / 180, 4. * np.pi / 180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 762.  # 1590
    coll_tilt_x = 0
    coll_tilt_y = 0
    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H2 = trace.to_next_surface(H_init, DC_slit, d_slit_col)
    rad = -762 * 2
    H3, DC3 = collimator.DCcoll(H2, DC_slit, T_coll, rad)

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -800)
    z_ech = d_col_ech + H3[:, 2]
    # print(z_ech)
    G = 44.41 * 1e-3  # lines/um
    # G = 1 / d
    H4 = trace.to_next_surface(H3, DC3, z_ech)

    ech_blaze = 70.  # deg
    ech_gamma = 4.  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H4, DC4 = echelle.diffraction(H4, DC3, T_echelle, order, wave, G)

    # todo bien hasta aca
    # tracing to cross-dispersing prisms
    d_ech_prm = np.full(len(H_init), 300)
    H5 = trace.to_next_surface(H4, DC4, d_ech_prm)
    # print(DC4)
    prisms_material = 'SF11'
    dec_x = 95.
    dec_y = 0.
    apex_angle = -34.6
    Tin = np.array([0.0 * np.pi / 180, 8. * np.pi / 180, 0.0 * np.pi / 180])
    H5, DC5 = prisms.tracing(H5, DC4, Tin, wave, prisms_material, apex_angle, dec_x, dec_y)
    H5 = np.array(H5)
    fcam = 300
    H6, DC6 = paraxial.tracing(H5, DC5, fcam, fcam)
    # print(H6)
    # print(H5, DC5)
    # Hout = H6.copy()
    # DCout = DC6.copy()
    ccd_decx = 0.4
    ccd_decy = -5.
    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['y'] = -H6[:, 0] + ccd_decx
    specout['x'] = H6[:, 1] + ccd_decy
    specout['z'] = H6[:, 2]
    specout['dx'] = DC6[:, 0]
    specout['dy'] = DC6[:, 1]
    specout['dz'] = DC6[:, 2]
    specout['flux'] = flux
    specout = CCD.mm2pix_custom(specout, det)

    return specout



if __name__ =='__main__':
    spectrum = echelle_orders.init()
    #specmoes = tracing(spectrum)
    #speczmx = pyzmx.fideos_tracing(18)
    #set_numpy_decimal_places(20, 0)
    #spectrum = echelle_orders.init_slit()pixelize_2D(0, 300)
    # print(spectrum)
    starpath = 'stellar_spectrum/'
    #spectrum = pd.read_csv(starpath+'stellar_slit_spec_rv_500ms.dat', sep=',')
    #print(spectrum)
    specmoes2 = raytrace(spectrum)

    #print(specmoes2)
    #specmoes2.to_csv(starpath+'moes_stellar_spectrum_rv_500ms.csv', index=False)
    #specmoes = open(starpath'')
    #print(3.74016645 -18.33066634)
    #print(3.05990931 -17.65040919)
    #print(specmoes)
    #plt.plot(speczmx[:, 0], speczmx[:, 1], 'ro')
    print(specmoes2.columns)
    specmoes2 = specmoes2.loc[specmoes2['x'] > 0]
    specmoes2 = specmoes2.loc[specmoes2['x'] < 2048]
    specmoes2 = specmoes2.loc[specmoes2['y'] > 0]
    specmoes2 = specmoes2.loc[specmoes2['y'] < 2048]

    #specmoes2 = specmoes2.loc[:, ~(specmoes2 == 105).any()]
    #print(np.unique(specmoes2['order']))
    #specmoes2 = specmoes2.loc[specmoes2['order'] != int(105)]
    #specmoes2 = specmoes2.loc[specmoes2['order'] != int(62)]
    specmoes2 = specmoes2.loc[specmoes2['order'] > 63]
    orders = np.unique(specmoes2['order'])
    maxorder = max(specmoes2['order'])
    minorder = min(specmoes2['order'])

    # print(specmoes2)
    print(maxorder, minorder)
    print(min(specmoes2['wave'].values), max(specmoes2['wave'].values))
    #import time
    #time.sleep(5)
    plt.plot(specmoes2['x'], specmoes2['y'], 'b.')
    plt.xlim(0, 2048)
    plt.ylim(0, 2048)

    plt.show()
    #print(70000+6500+10000+25000)
    #print(761.07077137-762)
    #print(speczmx)
    #print(specmoes)

    #plt.show()
    #ap = 1/(2*22)
    #print(ap)
    #print(2*762)
    #print(152/2)
    #print(spectrum)

