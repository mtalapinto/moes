import numpy as np


def get_name(i):
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    param_file = open(basedir+'init_params_siman_a.txt', 'r')
    params_names = []
    for line in param_file:
        linea = line.split()
        params_names.append(str(linea[0]))

    for k in range(len(params_names)):

        if k == i:
            param_name = params_names[k]

    return str(param_name)


def load():
    params = []
    # Load parameters
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    param_file = open(basedir+'init_params.txt', 'r')
    for line in param_file:
        linea = line.split()
        params.append(float(linea[2]))
    return params


def load4plots():
    params = []
    # Load parameters
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    param_file = open(basedir+'init_params_for_plots.txt', 'r')
    for line in param_file:
        linea = line.split()
        params.append(float(linea[2]))

    return params


def load_sa(date, fib):
    params = []
    # Load parameters
    basedir = '/luthien/carmenes/vis/params/'
    if fib == 'a':
        param_file = open(basedir+str(date)+'_init_params_siman_a.txt', 'r')
    else:
        param_file = open(basedir+str(date)+'_init_params_siman_b.txt', 'r')
    for line in param_file:
        linea = line.split()
        params.append(float(linea[2]))
    return params


def load_pymul():
    params = []
    # Load parameters
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    param_file = open(basedir+'init_params_pymul.txt', 'r')
    for line in param_file:
        linea = line.split()
        params.append(float(linea[2]))

    return params


def load_ini():
    params = []
    # Load parameters
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    param_file = open(basedir+'init_params.txt', 'r')
    for line in param_file:
        linea = line.split()
        params.append(float(linea[2]))

    return params


def adjust_err_bud(par, i):
    dT = 0.3  # K
    cte = np.array([
        0.0101*1e-6,  # Zerodur CTE in K-1
        20.3151*1e-6,      # Aluminum-5083 CTE
        9.1*1e-6,    # LF5 grismm in K-1
        8.4*1e-6,     # S-BAM4 glass in C-1
        13.1 * 1e-6,  # S-FPL51 glass in C-1
        14.5 * 1e-6,  # S-FPL53 glass in C-1
        7.2 * 1e-6,   # S-BSL7 glass in C-1
        6.1 * 1e-6,   # S-LAL10 glass in C-1
        0.51 * 1e-6,  # Silica glass in C-1
        8.1e-6,       # stim2
        11.1e-6,      # caf 2
        -0.23e-6,     # infrasil
        6.748e-6,     # sftm16
        4.6e-6        # znse


    ])

    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7:  # Slit parameters
        if par == 0.0:
            dx = cte[0] * dT
        else:
            dx = cte[0] * par * dT

    elif i == 8 or i == 15 or i == 18 or i == 26 or i == 30 or i == 36:   # distance parameters
        if par == 0.0:
            dx = cte[1] * dT
        else:
            dx = cte[1] * par * dT
    elif i == 11:       # echelle G
        daux = 1/par
        daux_dx = cte[0] * daux * dT
        dout = daux + daux_dx
        parout = 1/dout
        dx = np.abs(par - parout)
    elif i == 23:   # grism G
        daux = 1 / par
        daux_dx = cte[2] * daux * dT
        dout = daux + daux_dx
        parout = 1 / dout
        dx = np.abs(par - parout)
    elif i == 24:   # grism apex
        dx = cte[2]*par*dT
    else:   # angles an decenter
        if par == 0.0:
            dx = cte[1] * dT
        else:
            dx = cte[1] * par * dT

    return par, dx


def uniform_adjust_par(pars, i):
    '''
    if i == 2 or i == 4:
        dx = 0.1
    elif i == 11 or i == 12:
        dx = 0.01
    elif i == 19 or i == 20 or i == 23 or i == 24 or i == 28 or i == 29 or i == 33 or i == 34 or i == 35 or i == 36 or i == 37 or i == 38 or i == 39 or i == 40 or i == 41 or i == 42:
        dx = 0.1
    else:
        dx = 0.1
    '''
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4: # fiber decenter
        dx = 0.005
    elif i == 5 or i == 6 or i == 7:       # fiber angles
        dx = 0.001
    elif i == 8 or i == 15 or i == 18 or i == 26 or i == 30 or i == 36:  # distance parameters
        dx = 0.1  # in mm
    elif i == 19 or i == 20 or i == 26 or i == 27 or i == 26 or i == 27 or i == 31 or i == 32 or i == 37 or i == 38 or i == 39:
        dx = 0.1  # in mm
    elif i == 11 or i == 24:   # grating constants
        dx = 0.2
    else:           # angles
        dx = 0.1
    return pars, dx


def list_main_pars(par_sort):
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    file_in = open(basedir+'init_params.txt', 'r')
    file_out_0 = open(basedir+'params_list_ordered_uniform_y.dat', 'w')
    par_names = []

    for line in file_in:
        linea = line.split()
        par_names.append(linea[0])

    for i in range(len(par_sort)):

        par_aux = int(par_sort[i][0])
        for k in range(len(par_names)):
            if par_aux == k:
                file_out_0.write(par_names[k]+'\n')

    file_out_0.close()


def write(params):
    file = open(basedir+'init_params_crm.txt', 'w')
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
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
    return 'Parameters saved.'


def write_old(date, params, fib):

    basedir = '/luthien/carmenes/vis/params/old/'
    if fib == 'a':
        param_file = open(basedir + str(date) + '_init_params_siman_a.txt', 'w')
    else:
        param_file = open(basedir + str(date) + '_init_params_siman_b.txt', 'w')

    param_file.write(
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
    return 'Parameters saved'


def write_pymul(params):
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/parameters/'
    file = open(basedir+'init_params_pymul.txt','w')
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
    return 'Parameters saved.'


def write_sim(date, params, fib):
    outdir = '/luthien/carmenes/vis/params/'
    if fib == 'a':
        file = open(outdir+str(date)+'_init_params_siman_a.txt', 'w')
    if fib == 'b':
        file = open(outdir+str(date)+'_init_params_siman_b.txt', 'w')

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
    return 'Parameters saved...'


if __name__ == '__main__':

    pars = load()
    list_main_pars(pars)
