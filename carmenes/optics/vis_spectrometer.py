from . import slit
from . import fn_system
from . import collimator
from . import echelle
from . import flat_mirror
from . import grism
from . import camera
from . import field_flattener
from . import CCD_vis
from . import refraction_index
from . import trace
from . import transform
import numpy as np
from . import cte
from . import parameters
import pandas as pd


def tracing(spectrum, params, fib, temps):
    #
    # Variables initialization
    #
    
    temp_scaling = 1 # 0.99999 #1.0045

    H_init = np.zeros([len(spectrum), 3])
    DC_init = np.zeros([len(spectrum), 3])

    order = np.zeros(len(spectrum))
    wave = np.zeros(len(spectrum))
    #print(spectrum)
    order[:] = spectrum[:, 0]
    #order[:] = spectrum['order'].values
    #wave[:] = spectrum['wave'].values
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
    
    p = params[43]  # in Pa, 10e-5 in mbar
    temps_spec = temps[1:]
    t = np.average(temps_spec)
    wave = refraction_index.waves_air(wave, t, p)

    #
    # Slit data
    #
    
    if fib == 'A':
        slit_dec_x = np.full(len(spectrum), params[0])
        slit_dec_y = np.full(len(spectrum), params[1])
    elif fib == 'B':
        slit_dec_x = np.full(len(spectrum), params[2])
        slit_dec_y = np.full(len(spectrum), params[3])

    #
    # Position and initial orientation
    #
    
    defocus = params[4]
    H0, DC0 = slit.slit_params_init(H_init, DC_init, slit_dec_x, slit_dec_y, defocus)

    # To paraxial plane of the fn system
    d_fib_fn = 35.16
    H0 = trace.to_next_surface(H0, DC0, d_fib_fn)

    T_fib = np.asarray([0. * np.pi / 180, 0. * np.pi / 180, 0. * np.pi / 180])
    
    #
    # FN system
    #
    
    t_fn = temps[6]
    fndata = fn_system.load_data()
    fn_system_data = fn_system.set_data(fndata)
    H0, DC0 = fn_system.tracing(H0, DC0, T_fib, wave, t_fn, p, fn_system_data)
    T_slit = np.asarray([params[5]*np.pi/180, params[6]*np.pi/180, params[7]*np.pi/180])
    H1 = transform.transform(H0, -T_slit)
    DC1 = transform.transform(DC0, -T_slit)

    x.append(H1[:, 2])
    y.append(H1[:, 1])
    z.append(H1[:, 0])
    
    #
    # Collimator 1st pass
    #
    
    t_bench = temps[3]
    z_pos_col = params[8]  # 1590
    z_pos_col = temp_scaling*cte.recalc(z_pos_col, 'alum5083', t_bench)
    d_slit_col = np.abs(z_pos_col - H1[:, 2])
    t_coll_left = temps[2]
    t_coll_right = temps[1]
    t_coll = (t_coll_left + t_coll_right)/2
    coll_tilt_x = temp_scaling*cte.recalc(params[9], 'alum5083', t_coll)
    coll_tilt_y = temp_scaling*cte.recalc(params[10], 'alum5083', t_coll)
    T_coll = np.asarray([coll_tilt_x*np.pi/180, coll_tilt_y*np.pi/180, 0.*np.pi/180])
    H2 = trace.to_next_surface(H1, DC1, d_slit_col)
    curv_rad_aux = -1594.
    curv_rad_aux = temp_scaling*cte.recalc(curv_rad_aux, 'zerodur', t_coll_left)
    curvature_rad = np.full(len(H2), curv_rad_aux*2)
    H2, DC2 = collimator.DCcoll(H2, DC1, T_coll, curvature_rad)

    x.append(H2[:, 2])
    y.append(H2[:, 1])
    z.append(H2[:, 0])

    #
    # Echelle dispersion
    #
    d_col_ech_aux = -1594.305
    d_col_ech = temp_scaling*cte.recalc(d_col_ech_aux, 'alum5083', temps[5])
    z_pos_ech_aux = d_col_ech - d_col_ech_aux
    z_pos_ech = np.full(len(H_init), z_pos_ech_aux)
    H3 = trace.to_next_surface(H2, DC2, z_pos_ech)
    
    # Grating data
    G = params[11]*1e-3
    d = 1/G
    temp_echelle = (temps[6] + temps[8])/2
    d_new = temp_scaling*cte.recalc(d, 'zerodur', temp_echelle)
    G_new = 1/d_new

    # Orientation and diffraction
    ech_blaze = temp_scaling*cte.recalc(params[12], 'alum5083', temp_echelle)
    ech_gamma = temp_scaling*cte.recalc(params[13], 'alum5083', temp_echelle)
    ech_z_tilt = temp_scaling*cte.recalc(params[14], 'alum5083', temp_echelle)
    T_echelle = np.asarray([ech_blaze*np.pi/180, ech_gamma*np.pi/180, ech_z_tilt*np.pi/180])
    H3, DC3 = echelle.diffraction(H3, DC2, T_echelle, order, wave, G_new)
    
    #
    # Collimator 2nd pass
    #
    d_ech_col = np.full(len(H_init), z_pos_col)
    H4 = trace.to_next_surface(H3, DC3, d_ech_col)
    H4, DC4 = collimator.DCcoll(H4, DC3, T_coll, curvature_rad)
    
    #
    # Transfer mirror
    #
    d_col_tm_aux = params[15]
    d_col_tm = temp_scaling*cte.recalc(d_col_tm_aux, 'alum5083', t_bench)
    z_pos_tm_aux = d_col_ech - d_col_tm
    z_pos_tm = np.full(len(H_init), z_pos_tm_aux)
    H5 = trace.to_next_surface(H4, DC4, z_pos_tm)
    
    # Orientation
    tm_tilt_x = temp_scaling*cte.recalc(params[16], 'alum5083', temps[6])
    tm_tilt_y = temp_scaling*cte.recalc(params[17], 'alum5083', temps[6])
    T_flat = np.asarray([tm_tilt_x*np.pi/180, tm_tilt_y*np.pi/180, 0.0*np.pi/180])
    H5, DC5 = flat_mirror.flat_out(H5, DC4, T_flat)

    #
    # Collimator 3rd pass
    #
    
    d_trf_col = np.full(len(H_init), z_pos_col)
    H6 = trace.to_next_surface(H5, DC5, d_trf_col)
    curv_rad_aux = -1594.
    curv_rad_aux = temp_scaling*cte.recalc(curv_rad_aux, 'zerodur', t_coll_right)
    curvature_rad = np.full(len(H2), curv_rad_aux*2)
    H6, DC6 = collimator.DCcoll(H6, DC5, T_coll, curvature_rad)
    
    #
    # Grism
    #
    
    z_pos_grism = temp_scaling*cte.recalc(params[18], 'alum5083', temps[4])
    dcoll3_grism = np.full(len(H_init), z_pos_grism)
    H7 = trace.to_next_surface(H6, DC6, dcoll3_grism)

    # Position and orientation
    grm_dec_x = temp_scaling*cte.recalc(params[19], 'alum5083', temps[4])
    grm_dec_y = temp_scaling*cte.recalc(params[20], 'alum5083', temps[4])
    grism_dec_x = np.full(len(H_init), grm_dec_x)
    grism_dec_y = np.full(len(H_init), grm_dec_y)
    grm_tilt_x = temp_scaling*cte.recalc(params[21], 'alum5083', temps[4])
    grm_tilt_y = temp_scaling*cte.recalc(params[22], 'alum5083', temps[4])
    T_grism_in = np.asarray([grm_tilt_x * np.pi / 180, grm_tilt_y * np.pi / 180, 0. * np.pi / 180])

    # Material and grating data
    grism_material = 'LF5'
    dG = 1/(params[23] * 1e-3)
    dG_new = temp_scaling*cte.recalc(dG, 'lf5', temps[4])
    GD_new = 1/dG_new
    GD = np.full(len(H_init), GD_new)
    apex_grism = params[24]
    apex_grism = temp_scaling*cte.recalc(apex_grism, 'lf5', temps[4])
    H7, DC7 = grism.dispersion(H7, DC6, T_grism_in, wave, grism_material, apex_grism, GD, t, p, grism_dec_x, grism_dec_y)
    
    #
    #Camera
    #
    
    z_pos_cam = params[25]
    z_pos_cam = temp_scaling*cte.recalc(z_pos_cam, 'alum5083', temps[7])
    d_grism_cam = np.full(len(H_init), z_pos_cam)
    H8 = trace.to_next_surface(H7, DC7, d_grism_cam)

    # Position
    cam_dec_x = temp_scaling*cte.recalc(params[26], 'alum5083', temps[7])
    cam_dec_y = temp_scaling*cte.recalc(params[27], 'alum5083', temps[7])
    dec_x = np.full(len(H8), cam_dec_x)
    dec_y = np.full(len(H8), cam_dec_y)
    H8[:, 0] = H8[:, 0] + dec_x
    H8[:, 1] = H8[:, 1] + dec_y
    epx = H8[:, 0]
    epy = H8[:, 1]
    
    # Orientation
    cam_tilt_x = temp_scaling*cte.recalc(params[28], 'alum5083', temps[7])
    cam_tilt_y = temp_scaling*cte.recalc(params[29], 'alum5083', temps[7])
    T_cam = np.asarray([cam_tilt_x * np.pi / 180, cam_tilt_y * np.pi / 180, 0. * np.pi / 180])

    # Tracing camera lens 1 to 5
    d_cam_ff = temp_scaling*cte.recalc(params[30], 'alum5083', temps[7])
    camdata = camera.load_data()
    cam_data = camera.set_data(camdata)
    cam_data[-1][2] = d_cam_ff
    H8, DC8, H_cam_in = camera.tracing(H8, DC7, T_cam, wave, temps[7], p, cam_data)
    
    #
    # Field flattener
    #
    
    # position
    ccd_ff_dec_x = temp_scaling*cte.recalc(params[31], 'alum5083', temps[7])
    ccd_ff_dec_y = temp_scaling*cte.recalc(params[32], 'alum5083', temps[7])
    ff_dec_x = np.full(len(H_init), ccd_ff_dec_x, dtype='float64')
    ff_dec_y = np.full(len(H_init), ccd_ff_dec_y, dtype='float64')
    H8[:, 0] = H8[:, 0] + ff_dec_x
    H8[:, 1] = H8[:, 1] + ff_dec_y

    # orientation
    ccd_ff_tilt_x = temp_scaling*cte.recalc(params[33], 'alum5083', temps[7])
    ccd_ff_tilt_y = temp_scaling*cte.recalc(params[34], 'alum5083', temps[7])
    ccd_ff_tilt_z = temp_scaling*cte.recalc(params[35], 'alum5083', temps[7])
    T_ff_ccd = np.array([ccd_ff_tilt_x*np.pi/180, ccd_ff_tilt_y*np.pi/180, ccd_ff_tilt_z*np.pi/180])

    # Tracing
    ffdata = field_flattener.load_data()
    ff_data = field_flattener.set_data(ffdata)
    d_ff_ccd = temp_scaling*cte.recalc(params[36], 'alum5083', temps[7])
    ff_data[-1][2] = d_ff_ccd
    H9, DC9 = field_flattener.tracing(H8, DC8, T_ff_ccd, wave, temps[7], p, ff_data)
    Hff = H9.copy()
    # End Camera
    
    #
    # Detector
    #
    
    # Position
    temps_spec = np.average(temps[1:])

    t = np.average(temps_spec)
    ccd_dec_x = temp_scaling*cte.recalc(params[37], 'alum5083', t)
    ccd_dec_y = temp_scaling*cte.recalc(params[38], 'alum5083', t)
    ccd_defocus = temp_scaling*cte.recalc(params[39], 'alum5083', t)
    ccd_dec_x = np.full(len(H_init), ccd_dec_x, dtype='float64')
    ccd_dec_y = np.full(len(H_init), ccd_dec_y, dtype='float64')
    ccd_defocus = np.full(len(H_init), ccd_defocus, dtype='float64')
    H9[:, 0] = H9[:, 0] - ccd_dec_x
    H9[:, 1] = H9[:, 1] - ccd_dec_y
    H9[:, 2] = H9[:, 2] - ccd_defocus
    z_ff_ccd = ff_data[2][2]
    z_ff_ccd = temp_scaling*cte.recalc(z_ff_ccd, 'alum5083', t)
    H9 = trace.to_next_surface(H9, DC9, z_ff_ccd)
    H9[:, 2] = 0.

    # Orientation
    ccd_tilt_x = temp_scaling*cte.recalc(params[40], 'alum5083', t)
    ccd_tilt_y = temp_scaling*cte.recalc(params[41], 'alum5083', t)
    ccd_tilt_z = temp_scaling*cte.recalc(params[42], 'alum5083', t)
    T_ccd = np.array([ccd_tilt_x*np.pi/180, ccd_tilt_y*np.pi/180, ccd_tilt_z*180/np.pi])
    H9 = transform.transform(H9, -T_ccd)

    # Rotation to match with CARMENES frame geometry
    H9x_aux = H9[:, 0].copy()
    H9[:, 0] = -H9[:, 1]
    H9[:, 1] = H9x_aux
    Hff_aux = Hff[:, 0]
    Hff[:, 0] = -Hff[:, 1]
    Hff[:, 1] = Hff_aux
    ws = []
    
    for i in range(len(order)):
        ws.append([order[i], wave[i], H9[i][0], H9[i][1], H9[i][2], DC9[i][0], DC9[i][1], DC9[i][2], epx[i], epy[i]])

    ws = CCD_vis.mm2pix(np.asarray(ws))

    return ws

