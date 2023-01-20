def load(fib):
    params = []
    # Load parameters
    #basedir = '/luthien/carmenes/vis/params/'
    basedir = 'data/instrumental_parameters/'
    if fib == 'a':
        param_file = open(basedir+'init_params_siman_a.txt', 'r')
    else:
        param_file = open(basedir+'init_params_siman_b.txt', 'r')
    for line in param_file:
        linea = line.split()
        params.append(float(linea[2]))
    return params


def write(params, fib):
    outdir = 'data/instrumental_parameters/'
    if fib == 'a':
        file = open(outdir+'init_params_siman_a.txt', 'w')
    if fib == 'b':
        file = open(outdir+'init_params_siman_b.txt', 'w')

    file.write(
        'slit_dec_x_a = ' + str(params[0]) + '\n'
        'slit_dec_y_a = ' + str(params[1]) + '\n'
        'slit_dec_x_b = ' + str(params[2]) + '\n'
        'slit_dec_y_b = ' + str(params[3]) + '\n'
        'slit_defocus = ' + str(params[4]) + '\n'
        'slit_tilt_x = ' + str(params[5]) + '\n'
        'slit_tilt_y = ' + str(params[6]) + '\n'
        'slit_tilt_z = ' + str(params[7]) + '\n'
        'd_slit_col = ' + str(params[8]) + '\n'
        'coll_tilt_x = ' + str(params[9]) + '\n'
        'coll_tilt_y = ' + str(params[10]) + '\n'
        'ech_G = '+str(params[11])+'\n'
        'ech_blaze = '+str(params[12])+'\n'
        'ech_gamma = '+str(params[13])+'\n'
        'echelle_z = ' + str(params[14]) + '\n'
        'd_col_trf = ' + str(params[15]) + '\n'
        'trf_mirror_tilt_x = ' + str(params[16]) + '\n'
        'trf_mirror_tilt_y = ' + str(params[17]) + '\n'
        'd_col_grm = ' + str(params[18]) + '\n'
        'grism_dec_x = ' + str(params[19]) + '\n'
        'grism_dec_y = ' + str(params[20]) + '\n'
        'grm_tilt_x = ' + str(params[21]) + '\n'
        'grm_tilt_y = '+str(params[22])+'\n'
        'grm_G = '+str(params[23])+'\n'
        'grm_apex = '+str(params[24])+'\n'
        'd_grm_cam  = ' + str(params[25]) + '\n'                         
        'cam_dec_x = ' + str(params[26]) + '\n'
        'cam_dec_y = ' + str(params[27]) + '\n'
        'cam_tilt_x = ' + str(params[28]) + '\n'
        'cam_tilt_y = ' + str(params[29]) + '\n'
        'd_cam_ff  = ' + str(params[30]) + '\n'
        'ccd_ff_dec_x = ' + str(params[31]) + '\n'
        'ccd_ff_dec_y = ' + str(params[32]) + '\n'
        'ccd_ff_tilt_x = ' + str(params[33]) + '\n'
        'ccd_ff_tilt_y = ' + str(params[34]) + '\n'
        'ccd_ff_tilt_z = ' + str(params[35]) + '\n'
        'd_ff_ccd  = ' + str(params[36]) + '\n'
        'ccd_dec_x = ' + str(params[37]) + '\n'
        'ccd_dec_y = ' + str(params[38]) + '\n'
        'ccd_defocus = ' + str(params[39]) + '\n'
        'ccd_tilt_x = ' + str(params[40]) + '\n'
        'ccd_tilt_y = ' + str(params[41]) + '\n'
        'ccd_tilt_z = ' + str(params[42]) + '\n'           
        'p = ' + str(params[43])
    )
    print('Parameters saved.')
    return 'Parameters saved.'
