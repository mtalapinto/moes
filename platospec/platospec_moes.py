from optics import vph
from optics import CCD
from optics import flat_mirror
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
getcontext().prec = 20
from optics import refraction_index


def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})


def tracing(spectrum):  # (params, fib, temps):
    #
    # Variables initialization
    #

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
    DC_init[:, 2] = np.full(len(spectrum), 1)

    hx = 0.0
    hy = 0.0
    H_init[:, 0] = hx

    H_init[:, 1] = hy
    T_slit = np.asarray([0.*np.pi/180, 6*np.pi/180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 876.  # 1752
    coll_tilt_x = 0
    coll_tilt_y = 0

    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H_coll = trace.to_next_surface(H_init, DC_slit, d_slit_col)

    rad = -876. * 2
    H_coll, DC_coll = collimator.DCcoll(H_coll, DC_slit, T_coll, rad)
    H_coll[:, 2] = H_coll[:, 2] - z_pos_col

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -876)
    z_ech = d_col_ech + H_coll[:, 2]
    G = 41.6 * 1e-3  # lines/um
    H_ech = trace.to_next_surface(H_coll, DC_coll, z_ech)

    ech_blaze = -76.  # deg
    ech_gamma = 1.0  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H_ech, DC_ech = echelle.diffraction(H_ech, DC_coll, T_echelle, order, wave, G)

    # Collimator second pass
    d_ech_col = np.full(len(H_init), 876.)
    H_coll2 = trace.to_next_surface(H_ech, DC_ech, d_ech_col)
    # print(H5, DC4)
    H_coll2, DC_coll2 = collimator.DCcoll(H_coll2, DC_ech, T_coll, rad)
    H_coll2[:, 2] = H_coll2[:, 2] - z_pos_col
    DC_coll2 = -DC_coll2

    # Transfer mirror
    z_pos_tm = np.full(len(H_init), -876)
    H_tm = trace.to_next_surface(H_coll2, DC_coll2, z_pos_tm)

    # Orientation
    tm_tilt_x = 0.
    tm_tilt_y = 0
    T_flat = np.asarray([tm_tilt_x * np.pi / 180, tm_tilt_y * np.pi / 180, 0.0 * np.pi / 180])
    H_tm, DC_tm = flat_mirror.flat_out(H_tm, DC_coll2, T_flat)
    DC_tm = -DC_tm

    # Third collimation
    # To collimator
    z_pos_coll3 = np.full(len(H_init), 876)
    coll3_tilt_x = 0.
    coll3_tilt_y = 0.
    T_coll3 = np.asarray([coll3_tilt_x * np.pi / 180, coll3_tilt_y * np.pi / 180, 0.0 * np.pi / 180])
    H_coll3 = trace.to_next_surface(H_tm, DC_tm, z_pos_coll3)
    H_coll3, DC_coll3 = collimator.DCcoll(H_coll3, DC_tm, T_coll3, rad)
    H_coll3[:, 2] = H_coll3[:, 2] - z_pos_coll3
    DC_coll3 = -DC_coll3

    # VPH
    d_col_vph = np.full(len(H_init), -600)
    H_vph = trace.to_next_surface(H_coll3, DC_coll3, d_col_vph)
    H_vph[:, 2] = H_vph[:, 2] - d_col_vph

    vph_tilt_x = 0.
    vph_tilt_y = 0.
    vph_tilt_z = -90.
    Tvph = np.asarray([vph_tilt_x * np.pi / 180, vph_tilt_y * np.pi / 180, vph_tilt_z * np.pi / 180])
    H_vph, DC_vph = vph.tracing(H_vph, DC_coll3, wave, 'SILICA', Tvph)

    # Trace to camera
    d_vph_cam = np.full(len(H_vph), -100)
    H_cam = trace.to_next_surface(H_vph, DC_vph, d_vph_cam)
    H_cam[:, 2] = H_vph[:, 2] - d_vph_cam

    cam_tilt_x = 0.
    cam_tilt_y = -6.
    cam_tilt_z = 0.
    Tcam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, cam_tilt_z * np.pi / 180])
    H_cam = transform.transform2(H_cam, Tcam)
    DC_cam = transform.transform2(DC_vph, Tcam)
    H_cam[:, 0] = H_cam[:, 0] - (DC_cam[:, 0] / DC_cam[:, 2]) * (H_cam[:, 2])
    H_cam[:, 1] = H_cam[:, 1] - (DC_cam[:, 1] / DC_cam[:, 2]) * (H_cam[:, 2])
    H_cam[:, 2] = 0.

    H_ccd, DC_ccd = paraxial.tracing(H_cam, DC_cam, -135, -135)

    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = H_ccd[:, 1]
    specout['y'] = H_ccd[:, 0]
    specout['z'] = H_ccd[:, 2]
    specout['dx'] = DC_ccd[:, 0]
    specout['dy'] = DC_ccd[:, 1]
    specout['dz'] = DC_ccd[:, 2]

    return specout


def tracing_full(spectrum):  # (params, fib, temps):
    #
    # Variables initialization
    #
    set_numpy_decimal_places(20, 0)
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values

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
    print(H6)
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

    return specout


def tracing_full_fcam(spectrum, fcam):
    #
    # Variables initialization
    #
    vph_tilt_y = 0.
    GC = 340
    fcam = float(fcam)
    if fcam == 240.:
        ccd_decy = -27.4
    elif fcam == 250.:
        ccd_decy = -28.4
    elif fcam == 260.:
        ccd_decy = -29.4
    elif fcam == 270.:
        ccd_decy = -31.1
    elif fcam == 280:
        ccd_decy = -26.7
        vph_tilt_y = -0.8
        GC = 330

    set_numpy_decimal_places(20, 0)
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values
    flux = spectrum['flux'].values
    # Slit initialization
    H_init[:, 0] = spectrum['x'].values
    H_init[:, 1] = spectrum['y'].values
    H_init[:, 2] = spectrum['z'].values

    DC_init[:, 0] = spectrum['dx'].values
    DC_init[:, 1] = spectrum['dy'].values
    DC_init[:, 2] = spectrum['dz'].values

    T_slit = np.asarray([0.*np.pi/180, 6*np.pi/180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 876.  # 1752
    coll_tilt_x = 0
    coll_tilt_y = 0

    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H_coll = trace.to_next_surface(H_init, DC_slit, d_slit_col)

    rad = -876. * 2
    H_coll, DC_coll = collimator.DCcoll(H_coll, DC_slit, T_coll, rad)
    H_coll[:, 2] = H_coll[:, 2] - z_pos_col

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -876)
    z_ech = d_col_ech + H_coll[:, 2]
    G = 41.6 * 1e-3  # lines/um
    H_ech = trace.to_next_surface(H_coll, DC_coll, z_ech)

    ech_blaze = -76.  # deg
    ech_gamma = 1.0  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H_ech, DC_ech = echelle.diffraction(H_ech, DC_coll, T_echelle, order, wave, G)

    # Collimator second pass
    d_ech_col = np.full(len(H_init), 876.)
    H_coll2 = trace.to_next_surface(H_ech, DC_ech, d_ech_col)
    # print(H5, DC4)
    H_coll2, DC_coll2 = collimator.DCcoll(H_coll2, DC_ech, T_coll, rad)
    H_coll2[:, 2] = H_coll2[:, 2] - z_pos_col
    DC_coll2 = -DC_coll2

    # Transfer mirror
    z_pos_tm = np.full(len(H_init), -876)
    H_tm = trace.to_next_surface(H_coll2, DC_coll2, z_pos_tm)

    # Orientation
    tm_tilt_x = 0.
    tm_tilt_y = 0
    T_flat = np.asarray([tm_tilt_x * np.pi / 180, tm_tilt_y * np.pi / 180, 0.0 * np.pi / 180])
    H_tm, DC_tm = flat_mirror.flat_out(H_tm, DC_coll2, T_flat)
    DC_tm = -DC_tm

    # Third collimation
    # To collimator
    z_pos_coll3 = np.full(len(H_init), 876)
    coll3_tilt_x = 0.
    coll3_tilt_y = 0.
    T_coll3 = np.asarray([coll3_tilt_x * np.pi / 180, coll3_tilt_y * np.pi / 180, 0.0 * np.pi / 180])
    H_coll3 = trace.to_next_surface(H_tm, DC_tm, z_pos_coll3)
    H_coll3, DC_coll3 = collimator.DCcoll(H_coll3, DC_tm, T_coll3, rad)
    H_coll3[:, 2] = H_coll3[:, 2] - z_pos_coll3
    DC_coll3 = -DC_coll3

    # VPH
    d_col_vph = np.full(len(H_init), -600)
    H_vph = trace.to_next_surface(H_coll3, DC_coll3, d_col_vph)
    H_vph[:, 2] = H_vph[:, 2] - d_col_vph

    vph_tilt_x = 0.
    vph_tilt_z = -90.
    Tvph = np.asarray([vph_tilt_x * np.pi / 180, vph_tilt_y * np.pi / 180, vph_tilt_z * np.pi / 180])
    H_vph, DC_vph = vph.tracing(H_vph, DC_coll3, wave, 'SILICA', Tvph, GC)

    # Trace to camera
    d_vph_cam = np.full(len(H_vph), -100)
    H_cam = trace.to_next_surface(H_vph, DC_vph, d_vph_cam)
    H_cam[:, 2] = H_vph[:, 2] - d_vph_cam

    cam_tilt_x = 0.
    cam_tilt_y = -6.
    cam_tilt_z = 0.
    Tcam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, cam_tilt_z * np.pi / 180])
    H_cam = transform.transform2(H_cam, Tcam)
    DC_cam = transform.transform2(DC_vph, Tcam)
    H_cam[:, 0] = H_cam[:, 0] - (DC_cam[:, 0] / DC_cam[:, 2]) * (H_cam[:, 2])
    H_cam[:, 1] = H_cam[:, 1] - (DC_cam[:, 1] / DC_cam[:, 2]) * (H_cam[:, 2])
    H_cam[:, 2] = 0.

    H_ccd, DC_ccd = paraxial.tracing(H_cam, DC_cam, -fcam, -fcam)

    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = H_ccd[:, 1]
    specout['y'] = H_ccd[:, 0]
    specout['z'] = H_ccd[:, 2]
    specout['dx'] = DC_ccd[:, 0]
    specout['dy'] = DC_ccd[:, 1]
    specout['dz'] = DC_ccd[:, 2]

    ccd_decx = -1.6
    #ccd_decy = -30.4

    specout['x'] = specout['x'] + ccd_decx
    specout['y'] = -specout['y'] + ccd_decy
    det = [0, 15., 2048, 2048, 0]
    specout = CCD.mm2pix_custom(specout, det)
    specout['flux'] = flux

    return specout


def tracing_full_det(spectrum, det):
    #
    # Variables initialization
    #
    set_numpy_decimal_places(20, 0)
    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values
    flux = spectrum['flux'].values
    # Slit initialization
    H_init[:, 0] = spectrum['x'].values
    H_init[:, 1] = spectrum['y'].values
    H_init[:, 2] = spectrum['z'].values

    DC_init[:, 0] = spectrum['dx'].values
    DC_init[:, 1] = spectrum['dy'].values
    DC_init[:, 2] = spectrum['dz'].values

    T_slit = np.asarray([0.*np.pi/180, 6*np.pi/180, 0])
    DC_slit = transform.transform(DC_init, T_slit)

    # Collimator
    z_pos_col = 876.  # 1752
    coll_tilt_x = 0
    coll_tilt_y = 0

    d_slit_col = z_pos_col
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H_coll = trace.to_next_surface(H_init, DC_slit, d_slit_col)

    rad = -876. * 2
    H_coll, DC_coll = collimator.DCcoll(H_coll, DC_slit, T_coll, rad)
    H_coll[:, 2] = H_coll[:, 2] - z_pos_col

    # Echelle grating tracing
    d_col_ech = np.full(len(H_init), -876)
    z_ech = d_col_ech + H_coll[:, 2]
    G = 41.6 * 1e-3  # lines/um
    H_ech = trace.to_next_surface(H_coll, DC_coll, z_ech)

    ech_blaze = -76.  # deg
    ech_gamma = 1.0  # deg
    ech_z_tilt = 0.  # deg
    T_echelle = np.asarray([ech_blaze * np.pi / 180, ech_gamma * np.pi / 180, ech_z_tilt * np.pi / 180])
    H_ech, DC_ech = echelle.diffraction(H_ech, DC_coll, T_echelle, order, wave, G)

    # Collimator second pass
    d_ech_col = np.full(len(H_init), 876.)
    H_coll2 = trace.to_next_surface(H_ech, DC_ech, d_ech_col)
    # print(H5, DC4)
    H_coll2, DC_coll2 = collimator.DCcoll(H_coll2, DC_ech, T_coll, rad)
    H_coll2[:, 2] = H_coll2[:, 2] - z_pos_col
    DC_coll2 = -DC_coll2

    # Transfer mirror
    z_pos_tm = np.full(len(H_init), -876)
    H_tm = trace.to_next_surface(H_coll2, DC_coll2, z_pos_tm)

    # Orientation
    tm_tilt_x = 0.
    tm_tilt_y = 0
    T_flat = np.asarray([tm_tilt_x * np.pi / 180, tm_tilt_y * np.pi / 180, 0.0 * np.pi / 180])
    H_tm, DC_tm = flat_mirror.flat_out(H_tm, DC_coll2, T_flat)
    DC_tm = -DC_tm

    # Third collimation
    # To collimator
    z_pos_coll3 = np.full(len(H_init), 876)
    coll3_tilt_x = 0.
    coll3_tilt_y = 0.
    T_coll3 = np.asarray([coll3_tilt_x * np.pi / 180, coll3_tilt_y * np.pi / 180, 0.0 * np.pi / 180])
    H_coll3 = trace.to_next_surface(H_tm, DC_tm, z_pos_coll3)
    H_coll3, DC_coll3 = collimator.DCcoll(H_coll3, DC_tm, T_coll3, rad)
    H_coll3[:, 2] = H_coll3[:, 2] - z_pos_coll3
    DC_coll3 = -DC_coll3

    # VPH
    d_col_vph = np.full(len(H_init), -600)
    H_vph = trace.to_next_surface(H_coll3, DC_coll3, d_col_vph)
    H_vph[:, 2] = H_vph[:, 2] - d_col_vph

    vph_tilt_x = 0.
    vph_tilt_y = 0.
    vph_tilt_z = -90.
    Tvph = np.asarray([vph_tilt_x * np.pi / 180, vph_tilt_y * np.pi / 180, vph_tilt_z * np.pi / 180])
    H_vph, DC_vph = vph.tracing(H_vph, DC_coll3, wave, 'SILICA', Tvph)

    # Trace to camera
    d_vph_cam = np.full(len(H_vph), -100)
    H_cam = trace.to_next_surface(H_vph, DC_vph, d_vph_cam)
    H_cam[:, 2] = H_vph[:, 2] - d_vph_cam

    cam_tilt_x = 0.
    cam_tilt_y = -6.
    cam_tilt_z = 0.
    Tcam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, cam_tilt_z * np.pi / 180])
    H_cam = transform.transform2(H_cam, Tcam)
    DC_cam = transform.transform2(DC_vph, Tcam)
    H_cam[:, 0] = H_cam[:, 0] - (DC_cam[:, 0] / DC_cam[:, 2]) * (H_cam[:, 2])
    H_cam[:, 1] = H_cam[:, 1] - (DC_cam[:, 1] / DC_cam[:, 2]) * (H_cam[:, 2])
    H_cam[:, 2] = 0.
    fcam = 240.
    H_ccd, DC_ccd = paraxial.tracing(H_cam, DC_cam, -fcam, -fcam)

    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    specout['order'] = order
    specout['wave'] = wave
    specout['x'] = H_ccd[:, 1]
    specout['y'] = H_ccd[:, 0]
    specout['z'] = H_ccd[:, 2]
    specout['dx'] = DC_ccd[:, 0]
    specout['dy'] = DC_ccd[:, 1]
    specout['dz'] = DC_ccd[:, 2]

    ccd_decx = -1.6
    #ccd_decy = -30.4
    ccd_decy = -27.4
    specout['x'] = specout['x'] + ccd_decx
    specout['y'] = -specout['y'] + ccd_decy
    specout = CCD.mm2pix_custom(specout, det)
    specout['flux'] = flux

    return specout


if __name__ =='__main__':

    spectrum = echelle_orders.init_slit()
    #print(spectrum)
    set_numpy_decimal_places(20, 0)
    fcam = 280.
    fcol = 876.

    slit = 100
    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    pixsize = 13.5
    samp = fcam / fcol * slit / pixsize
    x_pix = x_um / pixsize
    y_pix = y_um / pixsize
    i = 0
    det = [samp, pixsize, x_pix, y_pix, i]
    specout = tracing_full_fcam(spectrum, fcam)
    specout = specout.loc[specout['x'] >= 0]
    specout = specout.loc[specout['x'] <= 2048]
    specout = specout.loc[specout['y'] >= 0]
    specout = specout.loc[specout['y'] <= 2048]
    #specout = specout.loc[specout['order'] == 73.]
    orders = np.unique(specout['order'])
    minorder = min(orders)
    maxorder = max(orders)
    print(minorder, maxorder)
    wmin = min(specout['wave'].values)
    wmax = max(specout['wave'].values)
    print(wmin, wmax)
    '''
    orders = np.unique(specout['order'])
    minorder = min(orders)
    maxorder = max(orders)
    simdir = 'data/pix_exp2/ccd_5/0/'
    omin = 68
    omax = 122
    while omin <= omax:
        data = pd.read_csv(simdir+str(int(omin))+'_2D_moes.tsv',sep=',')
        plt.plot(data['wave'], data['pix'], 'ro', alpha=0.5)
        omin += 1
    #plt.xlim(0, 2048)
    #plt.ylim(0, 2048)
    '''
    plt.plot(specout['x'], specout['y'], 'k.', markersize = 1)
    plt.show()
    plt.clf()

    x_um = 2048 * 13.5
    y_um = 2048 * 13.5
    '''
    pixarray = np.arange(7.5, 18, 1.5)
    fcam = 240.
    fcol = 876.
    slit = 100
    for pixsize in pixarray:
        samp = fcam / fcol * slit / pixsize
        x_pix = x_um / pixsize
        y_pix = y_um / pixsize
        print(samp, x_pix, y_pix)
        det = [pixsize, x_pix, y_pix]
        specout = tracing_full_det(spectrum, det)
        plt.clf()
        plt.xlim(0, int(x_pix) + 1)
        plt.ylim(0, int(y_pix) + 1)
        plt.plot(specout['x'], specout['y'], 'b.', markersize=1)
        plt.show()
        plt.clf()
    '''




