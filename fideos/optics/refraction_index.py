import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

def n_sell(l0, SC):
    # SC: Sellmeier coefficients
    n = np.sqrt(1 + SC[0]*l0**2/(l0**2 - SC[3]) + SC[1]*l0**2/(l0**2 - SC[4]) + SC[2]*l0**2/(l0**2 - SC[5]))
    return n
    

def SC(material):

    file = open('optics/materials.txt', 'r')
    file.readline()
    for line in file:
        
        SCs = line.split()
        
        if str(material) == str(SCs[0]):
            
            SC_out = np.asarray([float(SCs[1]), float(SCs[2]), float(SCs[3]), float(SCs[4]), float(SCs[5]), float(SCs[6])])
            file.close()
            return SC_out


def GC(material):
    file = open('optics/mat_t_coeffs.txt', 'r')
    file.readline()
    for line in file:
        GCs = line.split()
        if str(material) == str(GCs[0]):
            GC_out = np.asarray([float(GCs[1]), float(GCs[2]), float(GCs[3]), float(GCs[4]), float(GCs[5]), float(GCs[6])])
            file.close()
            return GC_out


def delta_n(lrel, nrel, GCs, t, tref):

    deltat = t - tref
    deltan = (nrel**2 - 1)*(GCs[0]*deltat + GCs[1]*deltat**2 + GCs[2]*deltat**3 + (GCs[3]*deltat + GCs[4]*deltat**2)/(lrel**2 - GCs[5]**2))/(2*nrel)
    return deltan


# x_CO2 in umol/mol
# T in Celsius, p in pascal
# f_tp no clue about what is it.
def waves_air(l, T, P):

    P = P*1e2 # from mbar to Pa
    w0 = 295.235
    w1 = 2.6422
    w2 = -0.03238
    w3 = 0.004028
    
    k0 = 238.0185
    k1 = 5792105
    k2 = 57.362
    k3 = 167917.0
    
    a0 = 1.58123e-6
    a1 = -2.9331e-8
    a2 = 1.1043e-10
    
    b0 = 5.707e-6
    b1 = -2.051e-8
    
    c0 = 1.9898e-4
    c1 = -2.376e-6
    
    d = 1.83e-11
    e = -0.765e-8
    
    p_R1 = 101325.0
    T_R1 = 288.15
    Z_a = 0.9995922115
    rho_vs = 0.00985938
    R = 8.314472
    M_v = 0.018015
    
    K1 = 1.16705214528e3
    K2 = -7.24213167032e5
    K3 = -1.70738469401e1
    K4 = 1.20208247025e4
    K5 = -3.23255503223e6
    K6 = 1.49151086135e1
    K7 = -4.82326573616e3
    K8 = 4.05113405421e5
    K9 = -2.38555575678e-1
    K10 = 6.50175348448e2

    S = 1 / (l ** 2)
    r_as = 1e-8*(k1/(k0 - S) + k3/(k2 - S))
    r_vs = 1.022e-8*(w0 + w1*S + w2*S**2 + w3*S**3)

    x_CO2 = 100
    M_a = 0.0289635 + 1.2011e-8*(x_CO2 - 450)
    r_axs = r_as*(1 + 5.34e-7*(x_CO2 - 450))
    t = T + 273.5

    Sigma = t + K9 / (t - K10)
    A = Sigma ** 2 + K1 * Sigma + K2
    B = K3 * Sigma ** 2 + K4 * Sigma + K5
    C = K6 * Sigma ** 2 + K7 * Sigma + K8
    X = -B + np.sqrt(B ** 2 - 4 * A * C)

    f_tp = 1
    RH = 1e-1

    p_sv = 10**6*(2*C/X)**4
    chi_v = (RH/100)*f_tp*p_sv/P
    Z_m = 1. - (P / T) * (a0 + a1 * t + a2 * t ** 2 + (b0 + b1 * t) * chi_v ** 2) + (P / T) ** 2 * (d + e * chi_v ** 2)
    rho_axs = p_R1 * M_a / (Z_a * R * T_R1)
    rho_v = chi_v * P * M_v / (Z_m * R * T)
    rho_a = (1. - chi_v) * P * M_a / (Z_m * R * T)
    n_air = 1. + (rho_a / rho_axs) * r_axs + (rho_v / rho_vs) * r_vs
    wav_air = l/n_air

    #for i in range(len(n_air)):
    #    print '%.15f %.15f\n' %(n_air[i], wav_air[i])

    return wav_air


def nref(l):
    return 1+(6432.8 + 2949810*l**2/(146*l**2 - 1) + 25540*l**2/(41*l**2 - 1))*1e-8


def nair_abs(l, T, P):
    return 1 + (nref(l) - 1)*P/(1 + 3.4785e-3*(T - 15))


def l_rel(l, nair_sys, nair_ref):
    return l*nair_sys/nair_ref


def n2(l, t, p, material):

    l_relative = waves_air(l, t, p)
    n_air_abs_sys = nair_abs(l_relative, t, p)

    if material == 'LF5':
        Tref = 20
    else:
        Tref = 25.  # in C

    Pref = 10.1325  # in Pa, 1 atm
    n_air_abs_ref = nair_abs(l_relative, Tref, Pref)
    l_rel = l*n_air_abs_sys/n_air_abs_ref
    sc = SC(material)
    n_rel_ref = n_sell(l_rel, sc)
    n_abs_ref = n_rel_ref*n_air_abs_ref
    gc = GC(material)
    if material == 'Air':
        n_abs_sys = nair_abs(l, t, p)
    else:
        n_abs_sys = n_abs_ref + delta_n(l_rel, n_rel_ref, gc, t, Tref)

    n_abs = n_abs_sys#/n_air_abs_sys

    return n_abs


def zmx(material):
    file_n = open('n_'+material+'_zmx.txt', 'r')
    n = []
    for line in file_n:
        linea = line.split()
        n.append(np.abs(float(linea[2])))
    return np.array(n)


def crm(material):
    path = '/home/guest/Dropbox/v21/vis/refractive_index/'
    file_n = open(path+'n_'+material+'_crm.txt', 'r')
    n = []
    for line in file_n:
        linea = line.split()
        n.append(np.abs(float(linea[2])))
    return np.array(n)


def n_final(l0, material):

    # path = '/home/guest/Dropbox/v21/vis/refractive_index/'
    path = '/home/marcelo/Dropbox/physical_modeling-code/v22/vis/refractive_index/'
    file = open(path + 'n_' + material + '_vis.txt', 'r')
    l = []
    n = []
    for line in file:
        linea = line.split()
        l.append(float(linea[0]))
        n.append(float(linea[1]))

    f = interp1d(l, n)
    #print l0
    return -f(l0)


def n(wave, t, p, material):

    waves_sys = waves_air(wave, t, p)

    if material == 'LF5' or material == 'SF11':
        tref = 20
    else:
        tref = 25.  # in C

    pref = 1.
    p = 1.
    t = 20.

    na_abs_sys = nair_abs(waves_sys, t, p)
    na_abs_ref = nair_abs(waves_sys, tref, pref)

    wave_rel = wave*na_abs_sys/na_abs_ref
    sc_lf5 = SC(material)
    ng_rel_ref = n_sell(wave_rel, sc_lf5)
    ng_abs_ref = ng_rel_ref*na_abs_ref
    gc_lf5 = GC(material)
    delta_n_abs = delta_n(wave_rel, ng_rel_ref, gc_lf5, t, tref)
    ng_abs_sys = ng_abs_ref + delta_n_abs
    ng_rel_sys = ng_abs_sys/na_abs_sys

    if material == 'Air':
        ng_rel_sys = na_abs_sys

    return ng_rel_sys


if __name__ == '__main__':

    file_temp = pd.read_csv('/home/marcelo/Documents/CARMENES/moes/v2.9/vis/env_data/VIS-CR-Ts01.dat', sep=',')
    file_temp2 = pd.read_csv('/home/marcelo/Documents/CARMENES/moes/v2.9/vis/env_data/VIS-CR-Ts02.dat', sep=',')
    waves_out_0 = waves_air(0.5, file_temp[' temp'].values, 1e-4)
    waves_out_02 = waves_air(0.5, file_temp2[' temp'].values, 1e-4)
    waves_out_1 = waves_air(0.7, file_temp[' temp'].values, 1e-4)
    waves_out_2 = waves_air(0.9, file_temp[' temp'].values, 1e-4)
    waves0 = (waves_out_0 - 0.5) * 3e8 / 0.5
    waves02 = (waves_out_02 - 0.5) * 3e8 / 0.5
    waves1 = (waves_out_1 - 0.7) * 3e8 / 0.7
    waves2 = (waves_out_2 - 0.9) * 3e8 / 0.9
    plt.plot(file_temp['mjd'].values, waves0, 'b+')
    plt.plot(file_temp2['mjd'].values, waves02, 'r+')


    #plt.plot(file_temp['mjd'].values, waves1, 'g+')
    #plt.plot(file_temp['mjd'].values, waves2, 'r+')

    plt.show()
    #print(waves_air(0.6, 25, 77000)*1e4)
    #print(waves_air(0.6, 25, 77500) * 1e4)
