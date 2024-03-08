import numpy as np
import copy


# sag equation for a cylindrical surface
def dZ(H, DCs, r):
    x_p = H[:, 0]
    y_p = H[:, 1]
    T_x = DCs[:, 0] / DCs[:, 2]
    T_y = DCs[:, 1] / DCs[:, 2]


    sign_r = r / np.abs(r)
    #dz = ((r - x_p * T_x) - sign_r*np.sqrt(r ** 2 - 2*r*x_p*T_x - x_p ** 2)) / (1 + T_x**2)
    dz = ((r - y_p * T_y) - sign_r * np.sqrt(r ** 2 - 2 * r * y_p * T_y - y_p ** 2)) / (1 + T_y ** 2)
    #dz = (T_x*x_p - r + np.sqrt((T_x*x_p - r)**2 - x_p - x_p*T_x**2))/(1 + T_x**2)

    H[:, 0] = H[:, 0] + dz * T_x
    H[:, 1] = H[:, 1] + dz * T_y
    H[:, 2] = dz


    #print DC_out[:, 0] ** 2 + DC_out[:, 1] ** 2 + DC_out[:, 2] ** 2

    cen = np.zeros([len(DCs), 3])
    cen[:, 0] = H[:, 0]
    cen[:, 1] = np.full(len(H), 0.)
    cen[:, 2] = np.full(len(H), r)
    sign_r = np.zeros([len(DCs), 3])
    sign_r[:, 0] = r / np.abs(r)
    sign_r[:, 1] = r / np.abs(r)
    sign_r[:, 2] = r / np.abs(r)
    sign_DCz = np.zeros([len(DCs), 3])
    sign_DCz[:, 0] = DCs[:, 2] / np.abs(DCs[:, 2])
    sign_DCz[:, 1] = DCs[:, 2] / np.abs(DCs[:, 2])
    sign_DCz[:, 2] = DCs[:, 2] / np.abs(DCs[:, 2])
    mod_n = np.zeros([len(DCs), 3])
    mod_n[:, 0] = np.sqrt((cen[:, 0] - H[:, 0]) ** 2 + (cen[:, 1] - H[:, 1]) ** 2 + (cen[:, 2] - H[:, 2]) ** 2)
    mod_n[:, 1] = np.sqrt((cen[:, 0] - H[:, 0]) ** 2 + (cen[:, 1] - H[:, 1]) ** 2 + (cen[:, 2] - H[:, 2]) ** 2)
    mod_n[:, 2] = np.sqrt((cen[:, 0] - H[:, 0]) ** 2 + (cen[:, 1] - H[:, 1]) ** 2 + (cen[:, 2] - H[:, 2]) ** 2)

    n = (cen - H) * sign_DCz * sign_r / mod_n

    return H, n







