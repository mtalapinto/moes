from . import transform
import copy
from . import cte
from . import spheric_surface
from . import refraction_index
import numpy as np
from . import trace


def tracing(H, DC, Tin, l0, t, p, fn_data):
    H[:, 2] = 0.

    #Orientation
    DC_out = transform.transform(DC, -Tin)
    H_out = transform.transform(H, -Tin)
    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.

    H_plane = copy.copy(H_out)

    # Lens 1
    # sf 0
    r_sf0 = fn_data[0][1]
    r_sf0 = cte.recalc(r_sf0, 'sfpl51', t)
    material_sf0 = fn_data[0][3]
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

    # sf0 - sf1
    z_l1_out = fn_data[0][2]
    z_l1_out = cte.recalc(z_l1_out, 'sfpl51', t)

    H_out = trace.to_next_surface(H_out, DC_out, z_l1_out)
    H_out[:, 2] = 0.

    # sf 1
    r_sf0 = fn_data[1][1]
    r_sf0 = cte.recalc(r_sf0, 'sfpl51', t)
    material_sf1 = fn_data[1][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)
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

    # Lens 1 - Lens 2

    z_l1_l2 = fn_data[1][2]
    z_l1_l2 = cte.recalc(z_l1_l2, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l1_l2)
    H_out[:, 2] = 0.

    # Lens 2
    r_sf0 = fn_data[2][1]
    r_sf0 = cte.recalc(r_sf0, 'stim2', t)
    material_sf0 = fn_data[2][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf0)

    # sf 0
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

    # sf 0 - sf 1
    z_l2 = fn_data[2][2]
    z_l2 = cte.recalc(z_l2, 'stim2', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_l2)
    H_out[:, 2] = 0

    # sf 1
    r_sf1 = fn_data[3][1]
    r_sf1 = cte.recalc(r_sf1, 'sfpl51', t)
    material_sf1 = fn_data[3][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf1)
    n0 = refraction_index.n(l0, t, p, material_sf0)
    n1 = refraction_index.n(l0, t, p, material_sf1)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # sf 1 - sf 2
    z_ff_sf2 = fn_data[3][2]
    z_ff_sf2 = cte.recalc(z_ff_sf2, 'sfpl51', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_ff_sf2)
    H_out[:, 2] = 0.

    # sf 2
    r_sf2 = fn_data[4][1]
    r_sf2 = cte.recalc(r_sf2, 'sfpl51', t)
    material_sf2 = fn_data[4][3]
    H_out, n = spheric_surface.dZ(H_out, DC_out, r_sf2)
    n0 = refraction_index.n(l0, t, p, material_sf1)
    n1 = refraction_index.nair_abs(l0, t, p)
    k = n1 / n0
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)
    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    z_fn_fp = fn_data[4][2]
    z_fn_fp = cte.recalc(z_fn_fp, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_fn_fp)
    H_out[:, 2] = 0.

    return H_out, DC_out


def set_data(fndata):
    
    fn_data = [
        [1, fndata[0], fndata[1], 'SFPL51-FN'],
        [2, fndata[2], fndata[3], 'Air'],
        [3, fndata[4], fndata[5], 'STIM2'],
        [4, fndata[6], fndata[7], 'SFPL51-FN'],
        [5, fndata[8], fndata[9], 'Air'],
    ]
    return fn_data


def init():
        
    fndata = [108.104, 4.,
              -27.736, 0.5,
              52.28, 2.5,
              14.453, 7,
              -48.519, 119.283]
    
    return fndata


def load_data():
    
    file_fn = open('optics/fn_data.dat','r')
    fndata = []
    for line in file_fn:
        fndata.append(float(line))

    return fndata


if __name__ == '__main__':
    
    fndata = init()
    file_fn = open('fn_data.dat','w')
    for i in range(len(fndata)):
        file_fn.write('%.8f\n' %(fndata[i]))
    file_fn.close()
