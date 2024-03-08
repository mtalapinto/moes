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
    basedir = '/home/eduspec/Documentos/moes/v3.1/vis/optics/'
    file_cam = open(basedir+'cam_data.dat','r')
    camdata = []
    for line in file_cam:
        camdata.append(float(line))

    return camdata


def tracing(H, DC, Tin, l0, t, p, cam_data):
    DC_out = np.zeros([len(DC), 3])
    H_out = np.zeros([len(H), 3])

    H[:, 2] = 0.

    #Orientation
    DC_out = transform.transform(DC, -Tin)
    H_out = transform.transform(H, -Tin)
    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.

    H_plane = copy.copy(H_out)

    # Lens 1

    # sf 19
    r_sf0 = cam_data[0][1]
    r_sf0 = cte.recalc(r_sf0, 'sfpl51', t)
    material_sf0 = cam_data[0][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)

    # n0 = np.full(len(H), 1.)
    n0 = refraction_index.nair_abs(l0, t, p)  # coming from air
    n1 = refraction_index.n(l0, t, p, material_sf0)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # sf0 - sf1
    z_l1_out = cam_data[0][2]
    z_l1_out = cte.recalc(z_l1_out, 'sfpl51', t)

    H_out = trace.to_next_surface(H_out, DC_out, z_l1_out)
    H_out[:, 2] = 0.

    # sf 20
    r_sf1 = cam_data[1][1]
    r_sf1 = cte.recalc(r_sf1, 'sfpl51', t)
    material_sf1 = cam_data[1][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf1)
    n0 = refraction_index.n(l0, t, p, material_sf0)
    n1 = refraction_index.nair_abs(l0, t, p)  # coming from air
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # End lens 1

    # Lens 1 - lens 2
    z_l2_in = cam_data[1][2]
    z_l2_in = cte.recalc(z_l2_in, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l2_in)
    H_out[:, 2] = 0.

    # Lens 2

    # sf 21
    r_sf0 = cam_data[2][1]
    r_sf0 = cte.recalc(r_sf0, 'sbam4', t)
    material_sf0 = cam_data[2][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)

    # n0 = np.full(len(H), 1.)
    n0 = refraction_index.nair_abs(l0, t, p)  # coming from air
    n1 = refraction_index.n(l0, t, p, material_sf0)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # sf 21 - sf 22
    z_l2_out = cam_data[2][2]
    z_l2_out = cte.recalc(z_l2_out, 'sbam4', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l2_out)
    H_out[:, 2] = 0.

    # sf 22
    r_sf1 = cam_data[3][1]
    r_sf1 = cte.recalc(r_sf1, 'alum5083', t)
    material_sf1 = cam_data[3][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf1)

    n0 = refraction_index.n(l0, t, p, material_sf0)
    n1 = refraction_index.nair_abs(l0, t, p)  # coming from air
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # End lens 2

    # Lens 2 - lens 3
    z_l3_in = cam_data[3][2]
    z_l3_in = cte.recalc(z_l3_in, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_in)
    H_out[:, 2] = 0.

    # Lens 3

    # sf 23
    r_sf0 = cam_data[4][1]
    r_sf0 = cte.recalc(r_sf0, 'sfpl53', t)
    material_sf0 = cam_data[4][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    # n0 = np.full(len(H), 1.)
    n0 = refraction_index.nair_abs(l0, t, p)  # coming from air
    n1 = refraction_index.n(l0, t, p, material_sf0)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # sf 23 - sf 24
    z_l3_out = cam_data[4][2]
    z_l3_out = cte.recalc(z_l3_out, 'sfpl53', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l3_out)
    H_out[:, 2] = 0.

    # sf 24
    r_sf1 = cam_data[5][1]
    r_sf1 = cte.recalc(r_sf1, 'sfpl53', t)
    material_sf1 = cam_data[5][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf1)

    n0 = refraction_index.n(l0, t, p, material_sf0)
    n1 = refraction_index.nair_abs(l0, t, p)  # coming from air
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]
    # End lens 3

    # Lens 3 - lens 4
    z_l4_in = cam_data[5][2]
    z_l4_in = cte.recalc(z_l4_in, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l4_in)
    H_out[:, 2] = 0.

    # Lens 4

    # surface 25
    r_sf0 = cam_data[6][1]
    r_sf0 = cte.recalc(r_sf0, 'sbsl7', t)
    material_sf0 = cam_data[6][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)

    n0 = refraction_index.nair_abs(l0, t, p)  # coming from air
    n1 = refraction_index.n(l0, t, p, material_sf0)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # surface 25 - 26
    z_l4_out = cam_data[6][2]
    z_l4_out = cte.recalc(z_l4_out, 'sbsl7', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l4_out)
    H_out[:, 2] = 0.

    # sf 26
    r_sf1 = cam_data[7][1]
    r_sf1 = cte.recalc(r_sf1, 'sbsl7', t)
    material_sf1 = cam_data[7][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf1)
    n0 = refraction_index.n(l0, t, p, material_sf0)
    n1 = refraction_index.nair_abs(l0, t, p)  # coming from air
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]
    # End lens 4

    # Lens 4 - lens 5
    z_l5_in = cam_data[7][2]
    z_l5_in = cte.recalc(z_l5_in, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l5_in)
    H_out[:, 2] = 0.

    # Lens 5

    # surface 27
    r_sf0 = cam_data[8][1]
    r_sf0 = cte.recalc(r_sf0, 'sbam4', t)
    material_sf0 = cam_data[8][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
    n0 = refraction_index.nair_abs(l0, t, p)  # coming from air
    n1 = refraction_index.n(l0, t, p, material_sf0)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # surface 27 - 28
    z_l5_out = cam_data[8][2]
    z_l5_out = cte.recalc(z_l5_out, 'sbam4', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l5_out)
    H_out[:, 2] = 0.

    # sf 28
    r_sf1 = cam_data[9][1]
    r_sf1 = cte.recalc(r_sf1, 'sbam4', t)
    material_sf1 = cam_data[9][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf1)
    n0 = refraction_index.n(l0, t, p, material_sf0)
    n1 = refraction_index.nair_abs(l0, t, p)  # coming from air
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]
    # End lens 5

    # Lens 5 - field flattener
    z_ff_in = cam_data[9][2]
    z_ff_in = cte.recalc(z_ff_in, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_ff_in)
    H_out[:, 2] = 0.
    # End camera one
    return H_out, DC_out, H_plane


def third_order_aberration_correction(ws_crm, pol_x, pol_y, x, y):
    res_x = (ws_crm[:, 2] - x)
    res_y = (ws_crm[:, 5] - y)
    sigma_x = la.lstsq(pol_x, res_x)[0]
    delta_x = np.dot(pol_x, sigma_x)

    sigma_y = la.lstsq(pol_y, res_y)[0]
    delta_y = np.dot(pol_y, sigma_y)

    return delta_x, delta_y


def third_order_aberration_poly(x, y):
    ep = 502 # mm
    x_prime = x/ep
    y_prime = y/ep

    max_x = max(np.abs(x))
    max_y = max(np.abs(y))

    if max_x < max_y:
        x = x/max_y
        y = y/max_y
    else:
        x = x / max_x
        y = y / max_x

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    ab_pol_y = np.array([r**3*np.cos(theta), (2+2*np.cos(2*theta))*r**2*y_prime, 3*r*y_prime**2*np.cos(theta),
                         r*y_prime**2*np.cos(theta), y_prime**3])
    ab_pol_x = np.array([r**3*np.sin(theta), r**2*y_prime*np.sin(2*theta), r*y_prime**2*np.sin(theta),
                         r*y_prime**2*np.sin(theta), np.zeros(len(x))])
    ab_pol_y = ab_pol_y.transpose()
    ab_pol_x = ab_pol_x.transpose()
    return ab_pol_x, ab_pol_y


def fifth_order_aberration_poly(x, y, x0, y0):
    max_x0 = max(np.abs(x0))
    max_y0 = max(np.abs(y0))

    if max_x0 < max_y0:
        x0 = x0/max_y0
        y0 = y0/max_y0
    else:
        x0 = x0 / max_x0
        y0 = y0 / max_x0

    detector_size = 4096*15*1e-3
    radio = detector_size/2
    x = x/radio
    y = y/radio
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan(y / x)
    ab_poly_y = np.array([np.cos(theta)*r**5,
                          r**4*y0,
                          r**4*y0*np.cos(2*theta),
                          np.cos(theta)*r**3*y0**2,
                          np.zeros(len(x)),
                          (np.cos(theta))**2*np.cos(theta) * r ** 3 * y0 ** 2,
                          r**2*y0**3,
                          np.cos(theta)*r**2*y0**3,
                          np.zeros(len(x)),
                          np.cos(theta)*r*y0**4,
                          np.zeros(len(x)),
                          y0**5
                          ])

    ab_poly_x = np.array([np.sin(theta) * r ** 5,
                          np.zeros(len(x)),
                          r ** 4 * x0 * np.sin(2 * theta),
                          np.zeros(len(x)),
                          np.sin(theta) * r ** 3 * x0 ** 2,
                          (np.cos(theta))**2*np.sin(theta) * r ** 3 * x0 ** 2,
                          np.zeros(len(x)),
                          np.zeros(len(x)),
                          np.sin(2*theta)*r**2*x0**3,
                          np.zeros(len(x)),
                          np.sin(theta)*r*x0**4,
                          np.zeros(len(x))
                          ])

    ab_pol_y = ab_poly_y.transpose()
    ab_pol_x = ab_poly_x.transpose()
    return ab_pol_x, ab_pol_y


def seidel_aberration_poly(H_ex_pup, H_ccd):
    x_ep = H_ex_pup[:, 0]/max(H_ex_pup[:, 0])
    y_ep = H_ex_pup[:, 1]/max(H_ex_pup[:, 1])
    tau_x = H_ccd[:, 0]/4096
    tau_y = H_ccd[:, 1]/4096
    r = np.sqrt(x_ep**2 + y_ep**2)
    theta = np.arctan(y_ep/x_ep)
    ab_pol_x = np.array([r**4, tau_x*r**3*np.cos(theta), tau_x**2*r**2*(np.cos(theta))**2,
                         tau_x**2*r ** 2, tau_x**3*r*np.cos(theta)])
    ab_pol_y = np.array([r ** 4, tau_y * r ** 3 * np.cos(theta), tau_y ** 2 * r ** 2 * (np.cos(theta)) ** 2,
                         tau_y ** 2 * r ** 2, tau_y ** 3 * r * np.cos(theta)])

    ab_pol_y = ab_pol_y.transpose()
    ab_pol_x = ab_pol_x.transpose()
    return ab_pol_x, ab_pol_y


def seidel_aberration_correction(pol_x, pol_y, x, y):
    #res_x = (ws_crmn[:, 3] - x)/4096
    #res_y = (ws_crmn[:, 5] - y)/4096
    #sigma_x = la.lstsq(pol_x, res_x)[0]
    #delta_x = np.dot(pol_x, sigma_x)

    #sigma_y = la.lstsq(pol_y, res_y)[0]
    #delta_y = np.dot(pol_y, sigma_y)
    delta_x = 0
    delta_y = 0
    return delta_x, delta_y


def load_data_original():
    cam_data = [
        [1, -299.99, -26.079, 'SFPL51-CAM'],
        [2, 1e10, -55.821, 'Air'],
        [3, -558.957, -11.65, 'SBAM4'],
        [4, -147.498, -0.995, 'Air'],
        [5, -147.491, -38.060, 'SFPL53'],
        [6, -2146.267, -19.695, 'Air'],
        [7, 495.253, -15.06, 'SBSL7'],
        [8, 999.741, -322.22, 'Air'],
        [9, -262.531, -25.98, 'SBAM4'],
        [10, 1097.82, -160.486, 'Air'],

    ]
    return cam_data


def load_data_alt():

    cam_data = [-299.99,
                -26.079,
                1e10,
                -55.821,
                -558.957,
                -11.65,
                -147.498,
                -0.995,
                -147.491,
                -38.060,
                -2146.267,
                -19.695,
                495.253,
                -15.06,
                999.741,
                -322.22,
                -262.531,
                -25.98,
                1097.82,
                -160.486
                ]

    return cam_data


def load_data_alt2():
    cam_data = [-302.82573181973567, -32.00798735680588, 10000000001.557877, -63.415910140923735, -559.113170464528, -14.00060438844059, -141.273425573828, -0.8165485184399665, -140.14256255539328, -33.884626021812316, -2144.658792500623, -14.710409957977339, 492.81043306360567, -13.65605613968947, 1000.7062692359641, -317.7782522015991, -259.14295278877387, -30.926088364171463, 1098.857252722748, -159.95079190047784]


    return cam_data
