from . import slit
from . import fn_system
from . import collimator
from . import echelle
from . import flat_mirror
from . import grism
from . import camera
from . import field_flattener
from . import CCD
from . import refraction_index
from . import trace
from . import transform
import numpy as np
from . import cte
from . import prism
from . import parameters
from . import paraxial
import pandas as pd
import matplotlib.pyplot as plt


def tracing_paraxial(spectrum): #, params, fib, temps):
    #
    # Variables initialization
    #
    

    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])
    Hout= np.zeros([len(spectrum), 3])
    DCout = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    order[:] = spectrum[:, 0]
    wave[:] = spectrum[:, 1]
    x = []
    y = []
    z = []

    H_init[:, 0] = np.zeros(len(spectrum))
    H_init[:, 1] = np.zeros(len(spectrum))
    H_init[:, 2] = np.zeros(len(spectrum))

    DC_init[:, 0] = np.zeros(len(spectrum))
    DC_init[:, 1] = np.zeros(len(spectrum))
    DC_init[:, 2] = np.zeros(len(spectrum))

    #
    # Environmental data
    #
    
    p = 1e0  # in Pa, 10e-5 in mbar
    temps_spec = [20.]
    t = np.average(temps_spec)
    #wave = refraction_index.waves_air(wave, t, p)

    #
    # Slit data
    #
    slit_dec_x = 0.0
    slit_dec_y = 0.0
    defocus = 0.0
    H0, DC0 = slit.slit_params_init(H_init, DC_init, slit_dec_x, slit_dec_y, defocus)
    #print(H0, DC0)

    # Tracing to collimator
    T0 = np.array([0, 6.44 * np.pi / 180, 0])
    DC1 = transform.transform(DC0, T0)
    d_slit_col = 1501
    H1 = trace.to_next_surface(H0, DC1, d_slit_col)

    # Collimator tracing
    coll_tilt_x = 0.0
    coll_tilt_y = 0.0
    coll_tilt_z = 0.0
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, coll_tilt_z * np.pi / 180])
    curvature_rad = -3002.
    H2, DC2 = collimator.DCcoll(H1, DC1, T_coll, curvature_rad)

    #
    # Echelle dispersion
    #

    d_col_ech = -1701.
    H3 = trace.to_next_surface(H2, DC2, d_col_ech)
    H3[:, 2] -= np.full(len(H3[:, 2]), d_col_ech)

    # Grating data
    G = 79. * 1e-3
    d = 1 / G
    T_echelle = np.asarray([63.4 * np.pi / 180, 0.6 * np.pi / 180, 0. * np.pi / 180])
    H3, DC3 = echelle.diffraction(H3, DC2, T_echelle, order, wave, G)

    # Tracing to collimator 1, 2nd pass
    d_ech_col = 1701.
    H4 = trace.to_next_surface(H3, DC3, d_ech_col)

    # Reflection at collimator
    H4, DC4 = collimator.DCcoll(H4, DC3, T_coll, curvature_rad)
    #DC4 = -DC4

    # Tracing to transfer mirror
    d_ech_col = -1501.
    H5 = trace.to_next_surface(H4, DC4, d_ech_col)

    # Reflection at transfer mirror
    T_flat = np.asarray([0. * np.pi / 180, 0. * np.pi / 180, 0.0 * np.pi / 180])
    H5, DC5 = flat_mirror.flat_out(H5, DC4, T_flat)

    # Tracing to collimator 2
    d_ech_col = 1601.
    H6 = trace.to_next_surface(H5, DC5, d_ech_col)

    # Reflection at collimator
    H6, DC6 = collimator.DCcoll(H6, DC5, T_coll, curvature_rad)

    # Tracing to cross-disperser
    d_col_cross = -1326.
    H7 = trace.to_next_surface(H6, DC6, d_col_cross)

    # Prism tracing
    prism_material = 'LF5'
    apex_prism = 55.
    prm_tilt_x = 0.0
    prm_tilt_y = 48.6
    prism_dec_x = -169.6
    prism_dec_y = 0.0
    T_grism_in = np.asarray([prm_tilt_x * np.pi / 180, prm_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H7, DC7 = prism.tracing(H7, DC6, T_grism_in, wave, prism_material, apex_prism, prism_dec_x, prism_dec_y)

    # tracing to camera
    d_prism_cam = -100.
    H8 = trace.to_next_surface(H7, DC7, d_prism_cam)

    #
    # Camera
    #
    # Position
    cam_dec_x = -109.
    cam_dec_y = 0.
    dec_x = np.full(len(H8), cam_dec_x)
    dec_y = np.full(len(H8), cam_dec_y)
    H8[:, 0] = H8[:, 0] - dec_x
    H8[:, 1] = H8[:, 1] - dec_y
    H8[:, 2] = np.zeros(len(H8))

    # Orientation
    cam_tilt_x = 0.
    cam_tilt_y = 47.2
    T_cam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, 0. * np.pi / 180])

    H8 = transform.transform2(H8, -T_cam)
    DC8 = transform.transform2(DC7, -T_cam)

    H8[:, 0] = H8[:, 0] - (DC8[:, 0] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 1] = H8[:, 1] - (DC8[:, 1] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 2] = 0.

    fcam = -410.
    H9, DC9 = paraxial.tracing(H8, DC8, fcam, fcam) #, cam_data)
    #print(H9)
    T_ccd = np.array([0. * np.pi/180, 0. * np.pi/180, 2.4 * np.pi/180])
    H9 = transform.transform2(H9, T_ccd)
    DC9 = transform.transform2(DC9, T_ccd)
    Hout, DCout = H9, DC9
    ws = []
    for i in range(len(order)):
        ws.append(np.array([order[i], wave[i], Hout[i][1], Hout[i][0], Hout[i][2], DCout[i][0], DCout[i][1], DCout[i][2]]))

    ws = np.array(ws)
    ws = CCD.mm2pix(ws)
    return ws


def tracing_paraxial_det(spectrum, det):  # , params, fib, temps):
    #
    # Variables initialization
    #

    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])
    Hout = np.zeros([len(spectrum), 3])
    DCout = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    flux = np.zeros(len(spectrum))
    order[:] = spectrum['order'].values
    wave[:] = spectrum['wave'].values
    flux[:] = spectrum['flux'].values


    H_init[:, 0] = np.zeros(len(spectrum))
    H_init[:, 1] = np.zeros(len(spectrum))
    H_init[:, 2] = np.zeros(len(spectrum))

    DC_init[:, 0] = np.zeros(len(spectrum))
    DC_init[:, 1] = np.zeros(len(spectrum))
    DC_init[:, 2] = np.zeros(len(spectrum))

    #
    # Environmental data
    #

    p = 1e0  # in Pa, 10e-5 in mbar
    temps_spec = [20.]
    t = np.average(temps_spec)
    # wave = refraction_index.waves_air(wave, t, p)

    #
    # Slit data
    #
    slit_dec_x = 0.0
    slit_dec_y = 0.0
    defocus = 0.0
    H0, DC0 = slit.slit_params_init(H_init, DC_init, slit_dec_x, slit_dec_y, defocus)
    # print(H0, DC0)

    # Tracing to collimator
    T0 = np.array([0, 6.44 * np.pi / 180, 0])
    DC1 = transform.transform(DC0, T0)
    d_slit_col = 1501
    H1 = trace.to_next_surface(H0, DC1, d_slit_col)

    # Collimator tracing
    coll_tilt_x = 0.0
    coll_tilt_y = 0.0
    coll_tilt_z = 0.0
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, coll_tilt_z * np.pi / 180])
    curvature_rad = -3002.
    H2, DC2 = collimator.DCcoll(H1, DC1, T_coll, curvature_rad)

    #
    # Echelle dispersion
    #

    d_col_ech = -1701.
    H3 = trace.to_next_surface(H2, DC2, d_col_ech)
    H3[:, 2] -= np.full(len(H3[:, 2]), d_col_ech)

    # Grating data
    G = 79. * 1e-3
    d = 1 / G
    T_echelle = np.asarray([63.4 * np.pi / 180, 0.6 * np.pi / 180, 0. * np.pi / 180])
    H3, DC3 = echelle.diffraction(H3, DC2, T_echelle, order, wave, G)

    # Tracing to collimator 1, 2nd pass
    d_ech_col = 1701.
    H4 = trace.to_next_surface(H3, DC3, d_ech_col)

    # Reflection at collimator
    H4, DC4 = collimator.DCcoll(H4, DC3, T_coll, curvature_rad)
    # DC4 = -DC4

    # Tracing to transfer mirror
    d_ech_col = -1501.
    H5 = trace.to_next_surface(H4, DC4, d_ech_col)

    # Reflection at transfer mirror
    T_flat = np.asarray([0. * np.pi / 180, 0. * np.pi / 180, 0.0 * np.pi / 180])
    H5, DC5 = flat_mirror.flat_out(H5, DC4, T_flat)

    # Tracing to collimator 2
    d_ech_col = 1601.
    H6 = trace.to_next_surface(H5, DC5, d_ech_col)

    # Reflection at collimator
    H6, DC6 = collimator.DCcoll(H6, DC5, T_coll, curvature_rad)

    # Tracing to cross-disperser
    d_col_cross = -1326.
    H7 = trace.to_next_surface(H6, DC6, d_col_cross)

    # Prism tracing
    prism_material = 'LF5'
    apex_prism = 55.
    prm_tilt_x = 0.0
    prm_tilt_y = 48.6
    prism_dec_x = -169.6
    prism_dec_y = 0.0
    T_grism_in = np.asarray([prm_tilt_x * np.pi / 180, prm_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H7, DC7 = prism.tracing(H7, DC6, T_grism_in, wave, prism_material, apex_prism, prism_dec_x, prism_dec_y)

    # tracing to camera
    d_prism_cam = -100.
    H8 = trace.to_next_surface(H7, DC7, d_prism_cam)

    #
    # Camera
    #
    # Position
    cam_dec_x = -109.
    cam_dec_y = 0.
    dec_x = np.full(len(H8), cam_dec_x)
    dec_y = np.full(len(H8), cam_dec_y)
    H8[:, 0] = H8[:, 0] - dec_x
    H8[:, 1] = H8[:, 1] - dec_y
    H8[:, 2] = np.zeros(len(H8))

    # Orientation
    cam_tilt_x = 0.
    cam_tilt_y = 47.2
    T_cam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, 0. * np.pi / 180])

    H8 = transform.transform2(H8, -T_cam)
    DC8 = transform.transform2(DC7, -T_cam)

    H8[:, 0] = H8[:, 0] - (DC8[:, 0] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 1] = H8[:, 1] - (DC8[:, 1] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 2] = 0.

    fcam = -410.
    H9, DC9 = paraxial.tracing(H8, DC8, fcam, fcam)  # , cam_data)
    # print(H9)
    T_ccd = np.array([0. * np.pi / 180, 0. * np.pi / 180, 2.4 * np.pi / 180])
    H9 = transform.transform2(H9, T_ccd)
    DC9 = transform.transform2(DC9, T_ccd)

    pd.set_option("display.precision", 20)
    specout = pd.DataFrame()
    decx = 17.2
    decy = -17.6
    specout['order'] = order
    specout['wave'] = wave
    specout['y'] = H9[:, 0] - decx
    specout['x'] = H9[:, 1] - decy
    specout['z'] = H9[:, 2]
    specout['dx'] = DC9[:, 0]
    specout['dy'] = DC9[:, 1]
    specout['dz'] = DC9[:, 2]
    specout['flux'] = flux
    specout = CCD.mm2pix_custom(specout, det)

    specout = specout.loc[specout['x'] <= det[2]]
    specout = specout.loc[specout['x'] >= 0.]
    specout = specout.loc[specout['y'] <= det[3]]
    specout = specout.loc[specout['y'] >= 0.]

    return specout


def tracing(spectrum):  # , params, fib, temps):
    #
    # Variables initialization
    #

    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])
    Hout = np.zeros([len(spectrum), 3])
    DCout = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    order[:] = spectrum['order']
    wave[:] = spectrum['wave']
    x = []
    y = []
    z = []

    H_init[:, 0] = np.zeros(len(spectrum))
    H_init[:, 1] = np.zeros(len(spectrum))
    H_init[:, 2] = np.zeros(len(spectrum))

    DC_init[:, 0] = np.zeros(len(spectrum))
    DC_init[:, 1] = np.zeros(len(spectrum))
    DC_init[:, 2] = np.zeros(len(spectrum))

    #
    # Environmental data
    #

    p = 1e0  # in Pa, 10e-5 in mbar
    temps_spec = [20.]
    t = np.average(temps_spec)
    # wave = refraction_index.waves_air(wave, t, p)

    #
    # Slit data
    #
    slit_dec_x = 0.0
    slit_dec_y = 0.0
    defocus = 0.0
    H0, DC0 = slit.slit_params_init(H_init, DC_init, slit_dec_x, slit_dec_y, defocus)
    # print(H0, DC0)

    # Tracing to collimator
    T0 = np.array([0, 6.44 * np.pi / 180, 0])
    DC1 = transform.transform(DC0, T0)
    d_slit_col = 1501
    H1 = trace.to_next_surface(H0, DC1, d_slit_col)

    # Collimator tracing
    coll_tilt_x = 0.0
    coll_tilt_y = 0.0
    coll_tilt_z = 0.0
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, coll_tilt_z * np.pi / 180])
    curvature_rad = -3002.
    H2, DC2 = collimator.DCcoll(H1, DC1, T_coll, curvature_rad)

    #
    # Echelle dispersion
    #

    d_col_ech = -1701.
    H3 = trace.to_next_surface(H2, DC2, d_col_ech)
    H3[:, 2] -= np.full(len(H3[:, 2]), d_col_ech)

    # Grating data
    G = 79. * 1e-3
    d = 1 / G
    T_echelle = np.asarray([63.4 * np.pi / 180, 0.6 * np.pi / 180, 0. * np.pi / 180])
    H3, DC3 = echelle.diffraction(H3, DC2, T_echelle, order, wave, G)

    # Tracing to collimator 1, 2nd pass
    d_ech_col = 1701.
    H4 = trace.to_next_surface(H3, DC3, d_ech_col)

    # Reflection at collimator
    H4, DC4 = collimator.DCcoll(H4, DC3, T_coll, curvature_rad)
    # DC4 = -DC4

    # Tracing to transfer mirror
    d_ech_col = -1501.
    H5 = trace.to_next_surface(H4, DC4, d_ech_col)

    # Reflection at transfer mirror
    T_flat = np.asarray([0. * np.pi / 180, 0. * np.pi / 180, 0.0 * np.pi / 180])
    H5, DC5 = flat_mirror.flat_out(H5, DC4, T_flat)

    # Tracing to collimator 2
    d_ech_col = 1601.
    H6 = trace.to_next_surface(H5, DC5, d_ech_col)

    # Reflection at collimator
    H6, DC6 = collimator.DCcoll(H6, DC5, T_coll, curvature_rad)

    # Tracing to cross-disperser
    d_col_cross = -1326.
    H7 = trace.to_next_surface(H6, DC6, d_col_cross)

    # Prism tracing
    prism_material = 'LF5'
    apex_prism = 55.
    prm_tilt_x = 0.0
    prm_tilt_y = 48.6
    prism_dec_x = -169.6
    prism_dec_y = 0.0
    T_grism_in = np.asarray([prm_tilt_x * np.pi / 180, prm_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H7, DC7 = prism.tracing(H7, DC6, T_grism_in, wave, prism_material, apex_prism, prism_dec_x, prism_dec_y)

    # tracing to camera
    d_prism_cam = -100.
    H8 = trace.to_next_surface(H7, DC7, d_prism_cam)
    #
    # Camera
    #
    # Position
    cam_dec_x = -109.
    cam_dec_y = 0.
    dec_x = np.full(len(H8), cam_dec_x)
    dec_y = np.full(len(H8), cam_dec_y)
    H8[:, 0] = H8[:, 0] - dec_x
    H8[:, 1] = H8[:, 1] - dec_y
    H8[:, 2] = np.zeros(len(H8))

    # Orientation
    cam_tilt_x = 0.
    cam_tilt_y = 47.2
    T_cam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, 0. * np.pi / 180])

    H8 = transform.transform2(H8, -T_cam)
    DC8 = transform.transform2(DC7, -T_cam)

    H8[:, 0] = H8[:, 0] - (DC8[:, 0] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 1] = H8[:, 1] - (DC8[:, 1] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 2] = 0.
    # Camera
    Hout, DCout = camera.tracing(H8, DC8, T_cam, wave, t, p)

    ws = []
    for i in range(len(order)):
        ws.append(
            np.array([order[i], wave[i], Hout[i][1], Hout[i][0], Hout[i][2], DCout[i][0], DCout[i][1], DCout[i][2]]))

    ws = np.array(ws)
    wsout = pd.DataFrame()
    wsout['order'] = ws[:, 0]
    wsout['wave'] = ws[:, 1]
    wsout['x'] = ws[:, 2]
    wsout['y'] = ws[:, 3]
    wsout['z'] = ws[:, 4]
    wsout['dx'] = ws[:, 5]
    wsout['dy'] = ws[:, 6]
    wsout['dz'] = ws[:, 7]

    return wsout


def tracing_det(spectrum, det):  # , params, fib, temps):
    #
    # Variables initialization
    #

    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])
    Hout = np.zeros([len(spectrum), 3])
    DCout = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    flux = np.zeros(len(spectrum))

    order[:] = spectrum['order']
    wave[:] = spectrum['wave']
    flux[:] = spectrum['flux']

    x = []
    y = []
    z = []

    H_init[:, 0] = spectrum['x'].values   #np.zeros(len(spectrum))
    H_init[:, 1] = spectrum['y'].values      #np.zeros(len(spectrum))
    H_init[:, 2] = np.zeros(len(spectrum))

    DC_init[:, 0] = np.zeros(len(spectrum))
    DC_init[:, 1] = np.zeros(len(spectrum))
    DC_init[:, 2] = np.zeros(len(spectrum))

    #
    # Environmental data
    #

    p = 1e0  # in Pa, 10e-5 in mbar
    temps_spec = [20.]
    t = np.average(temps_spec)
    # wave = refraction_index.waves_air(wave, t, p)

    #
    # Slit data
    #
    slit_dec_x = 0.0
    slit_dec_y = 0.0
    defocus = 0.0
    H0, DC0 = slit.slit_params_init(H_init, DC_init, slit_dec_x, slit_dec_y, defocus)
    # print(H0, DC0)

    # Tracing to collimator
    T0 = np.array([0, 6.44 * np.pi / 180, 0])
    DC1 = transform.transform(DC0, T0)
    d_slit_col = 1501
    H1 = trace.to_next_surface(H0, DC1, d_slit_col)

    # Collimator tracing
    coll_tilt_x = 0.0
    coll_tilt_y = 0.0
    coll_tilt_z = 0.0
    T_coll = np.asarray([coll_tilt_x * np.pi / 180, coll_tilt_y * np.pi / 180, coll_tilt_z * np.pi / 180])
    curvature_rad = -3002.
    H2, DC2 = collimator.DCcoll(H1, DC1, T_coll, curvature_rad)

    #
    # Echelle dispersion
    #

    d_col_ech = -1701.
    H3 = trace.to_next_surface(H2, DC2, d_col_ech)
    H3[:, 2] -= np.full(len(H3[:, 2]), d_col_ech)

    # Grating data
    G = 79. * 1e-3
    d = 1 / G
    T_echelle = np.asarray([63.4 * np.pi / 180, 0.6 * np.pi / 180, 0. * np.pi / 180])
    H3, DC3 = echelle.diffraction(H3, DC2, T_echelle, order, wave, G)

    # Tracing to collimator 1, 2nd pass
    d_ech_col = 1701.
    H4 = trace.to_next_surface(H3, DC3, d_ech_col)

    # Reflection at collimator
    H4, DC4 = collimator.DCcoll(H4, DC3, T_coll, curvature_rad)
    # DC4 = -DC4

    # Tracing to transfer mirror
    d_ech_col = -1501.
    H5 = trace.to_next_surface(H4, DC4, d_ech_col)

    # Reflection at transfer mirror
    T_flat = np.asarray([0. * np.pi / 180, 0. * np.pi / 180, 0.0 * np.pi / 180])
    H5, DC5 = flat_mirror.flat_out(H5, DC4, T_flat)

    # Tracing to collimator 2
    d_ech_col = 1601.
    H6 = trace.to_next_surface(H5, DC5, d_ech_col)

    # Reflection at collimator
    H6, DC6 = collimator.DCcoll(H6, DC5, T_coll, curvature_rad)

    # Tracing to cross-disperser
    d_col_cross = -1326.
    H7 = trace.to_next_surface(H6, DC6, d_col_cross)

    # Prism tracing
    prism_material = 'LF5'
    apex_prism = 55.
    prm_tilt_x = 0.0
    prm_tilt_y = 48.6
    prism_dec_x = -169.6
    prism_dec_y = 0.0
    T_grism_in = np.asarray([prm_tilt_x * np.pi / 180, prm_tilt_y * np.pi / 180, 0. * np.pi / 180])
    H7, DC7 = prism.tracing(H7, DC6, T_grism_in, wave, prism_material, apex_prism, prism_dec_x, prism_dec_y)

    # tracing to camera
    d_prism_cam = -100.
    H8 = trace.to_next_surface(H7, DC7, d_prism_cam)
    #
    # Camera
    #
    # Position
    cam_dec_x = -109.
    cam_dec_y = 0.
    dec_x = np.full(len(H8), cam_dec_x)
    dec_y = np.full(len(H8), cam_dec_y)
    H8[:, 0] = H8[:, 0] - dec_x
    H8[:, 1] = H8[:, 1] - dec_y
    H8[:, 2] = np.zeros(len(H8))

    # Orientation
    cam_tilt_x = 0.
    cam_tilt_y = 47.2
    T_cam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, 0. * np.pi / 180])

    H8 = transform.transform2(H8, -T_cam)
    DC8 = transform.transform2(DC7, -T_cam)

    H8[:, 0] = H8[:, 0] - (DC8[:, 0] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 1] = H8[:, 1] - (DC8[:, 1] / DC8[:, 2]) * (H8[:, 2])
    H8[:, 2] = 0.
    # Camera
    Hout, DCout = camera.tracing(H8, DC8, T_cam, wave, t, p)

    ws = []
    for i in range(len(order)):
        ws.append(
            np.array([order[i], wave[i], Hout[i][1], Hout[i][0], Hout[i][2], DCout[i][0], DCout[i][1], DCout[i][2]]))

    ws = np.array(ws)
    wsout = pd.DataFrame()
    wsout['order'] = ws[:, 0]
    wsout['wave'] = ws[:, 1]
    wsout['x'] = ws[:, 2]
    wsout['y'] = ws[:, 3]
    wsout['z'] = ws[:, 4]
    wsout['dx'] = ws[:, 5]
    wsout['dy'] = ws[:, 6]
    wsout['dz'] = ws[:, 7]
    wsout['flux'] = np.array(flux)
    wsout = CCD.mm2pix_custom(wsout, det)
    wsout['x'] = det[2] - wsout['x']
    wsout = wsout.loc[wsout['x'] > 0]
    wsout = wsout.loc[wsout['x'] <= det[2]]
    wsout = wsout.loc[wsout['y'] <= det[3]]
    wsout = wsout.loc[wsout['y'] > 0]
    return wsout
