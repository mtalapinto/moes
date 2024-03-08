from . import trace
from . import spheric_surface
import numpy as np
from . import transform
from . import refraction_index
import copy
import numpy.linalg as la
#import compare_zemax
#import matplotlib.pyplot as plt
from . import cte


def init():
    cam_data = [
        -299.99, -26.079,
        1e10, -55.821,
        -558.957, -11.65,
        -147.498, -0.995,
        -147.491, -38.060,
        -2146.267, -19.695,
        495.253, -15.06,
        999.741, -322.22,
        -262.531, -25.98,
        1097.82, -160.486
        ]

    return cam_data


def set_data(camdat):
    cam_data = [
        [1, camdat[0], camdat[1], 'SFPL51-CAM'],
        [2, camdat[2], camdat[3], 'Air'],
        [3, camdat[4], camdat[5], 'SBAM4'],
        [4, camdat[6], camdat[7], 'Air'],
        [5, camdat[8], camdat[9], 'SFPL53'],
        [6, camdat[10], camdat[11], 'Air'],
        [7, camdat[12], camdat[13], 'SBSL7'],
        [8, camdat[14], camdat[15], 'Air'],
        [9, camdat[16], camdat[17], 'SBAM4'],
        [10, camdat[18], camdat[19], 'Air']

    ]

    return cam_data


def load_data():
    basedir = 'optics/'
    file_cam = open(basedir+'cam_data.dat','r')
    camdata = []
    for line in file_cam:
        camdata.append(float(line))

    return camdata


def tracing(H, DC, Tin, l0, t, p):#, cam_data):
    H_out = H.copy()
    DC_out = DC.copy()

    # Lens 1 - entry surface
    r_sf0 = -230.664
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    n0 = np.full(len(DC), 1)
    n1 = refraction_index.n(l0, t, p, 'CAF2')

    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # sf0 - sf1
    z_l1_out = -24.7
    H_out = trace.to_next_surface(H_out, DC_out, z_l1_out)
    H_out[:, 2] = 0.

    # output surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, 633.82)
    n0 = refraction_index.n(l0, t, p, 'CAF2')
    n1 = np.full(len(DC), 1)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # End lens 1

    # Tracing to lens 2
    z_l2_out = -1
    H_out = trace.to_next_surface(H_out, DC_out, z_l2_out)
    H_out[:, 2] = 0.

    # Lens 2 - entry surface
    r_sf0 = -1766.220
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    n0 = np.full(len(DC), 1)
    n1 = refraction_index.n(l0, t, p, 'PSK3')

    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to lens 2, 2nd surface
    z_l3_out = -8.1
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # lens 2, 2nd surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, -130.030)
    n0 = refraction_index.n(l0, t, p, 'PSK3')
    n1 = refraction_index.n(l0, t, p, 'FK54')
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to lens 2, output surface
    z_l3_out = -30.070
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # lens 2, output surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, 1613.65)
    n0 = refraction_index.n(l0, t, p, 'FK54')
    n1 = np.full(len(DC), 1)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # End lens 2

    # Tracing to lens 3
    z_l2_out = -10
    H_out = trace.to_next_surface(H_out, DC_out, z_l2_out)
    H_out[:, 2] = 0.

    # Lens 3 - entry surface
    r_sf0 = 319.400
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    n0 = np.full(len(DC), 1)
    n1 = refraction_index.n(l0, t, p, 'BK7')
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to lens 3, 2nd surface
    z_l3_out = -15.030
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # lens 3, output surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, 803.7)
    n0 = refraction_index.n(l0, t, p, 'BK7')
    n1 = np.full(len(DC), 1)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to lens 4, 1nd surface
    z_l3_out = -314.07
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # Lens 4, 1st surface
    r_sf0 = -197.97
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    n0 = np.full(len(DC), 1)
    n1 = refraction_index.n(l0, t, p, 'FK5')
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to lens 4, 2nd surface
    z_l3_out = -25.030
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # lens 4, output surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, 1000.54)
    n0 = refraction_index.n(l0, t, p, 'FK5')
    n1 = np.full(len(DC), 1)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to lens 4, 1nd surface
    z_l3_out = -132.5
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # CCD rotation
    T_ccd = np.array([0.*np.pi/180, 0.*np.pi/180, 0.*np.pi/180])
    H_out = transform.transform2(H_out, T_ccd)
    DC_out = transform.transform2(DC_out, T_ccd)

    # Field lens, 1st surface
    r_sf0 = 135.01
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    n0 = np.full(len(DC), 1)
    n1 = refraction_index.n(l0, t, p, 'LAK16A')
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to FF, 2nd surface
    z_l3_out = -2.060
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # field lens 2nd surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, 1e20)
    n0 = refraction_index.n(l0, t, p, 'LAK16A')
    n1 = refraction_index.n(l0, t, p, 'SILICA')
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to field lens, next surface
    z_l3_out = -10.110
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # field lens 3rd surface
    H_out, n = spheric_surface.dZ(H_out, DC_out, 1e20)
    n0 = refraction_index.n(l0, t, p, 'SILICA')
    n1 = refraction_index.n(l0, t, p, 'VACUUM')
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # tracing to field lens, next surface
    z_l3_out = -4.7
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    return H_out, DC_out

