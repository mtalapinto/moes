from . import transform
from . import spheric_surface
from . import cylindric_surface
from . import refraction_index
import numpy as np
from . import trace
from . import cte


def load_data():

    file_ff = open('optics/field_flattener_data.dat','r')
    ffdat = []
    for line in file_ff:
        ffdat.append(float(line))

    return ffdat


def set_data(ffdat):

    #ffdat = [218.1727758412034, -7.068729635259748, 1e+30, -10.847020367291076, 530.508340450304, -4.879529128842498]
    ff_data = [[1, ffdat[0], ffdat[1], 'SLAL10'],
               [2, ffdat[2], ffdat[3], 'SILICA'],
               [3, ffdat[4], ffdat[5], 'Air']
               ]
    #ff = [[1, 218.1727758412034, -7.068729635259748, 'SLAL10'],
    #      [2, 1e30, -10.847020367291076, 'SILICA'],
    #      [3, 530.508340450304, -4.879529128842498, 'Air']
    #      ]

    return ff_data


def tracing(H, DC, Tin, l0, t, p, ff_data):

    H[:, 2] = 0.

    # Orientation
    DC_out = transform.transform(DC, -Tin)
    H_out = transform.transform(H, -Tin)
    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.


    # Surface 29
    r_sf0 = ff_data[0][1]
    r_sf0 = cte.recalc(r_sf0, 'slal10', t)
    material_sf0 = ff_data[0][3]
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

    # Surface 29 - 30
    z_ff_sf1 = ff_data[0][2]
    z_ff_sf1 = cte.recalc(z_ff_sf1, 'slal10', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_ff_sf1)
    H_out[:, 2] = 0.

    # Surface 30
    r_sf1 = ff_data[1][1]
    r_sf1 = cte.recalc(r_sf1, 'slal10', t)
    material_sf1 = ff_data[1][3]
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

    # Surface 30 - 31
    z_ff_sf2 = ff_data[1][2]
    z_ff_sf2 = cte.recalc(z_ff_sf2, 'alum5083', t)
    H_out = trace.to_next_surface(H_out, DC_out, z_ff_sf2)
    H_out[:, 2] = 0.


    # Surface 31
    r_sf2 = ff_data[2][1]
    r_sf2 = cte.recalc(r_sf2, 'silica', t)
    material_sf2 = ff_data[2][3]
    H_out, n = cylindric_surface.dZ(H_out, DC_out, r_sf2)
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

    # Field flattener to detector
    #z_ff_ccd = ff_data[2][2]
    #z_ff_ccd = cte.recalc(z_ff_ccd, 'alum5083', t)
    #H_out = trace.to_next_surface(H_out, DC_out, z_ff_ccd)
    #H_out[:, 2] = 0.

    return H_out, DC_out


def load_data_alt():

    #ffdata = [218.19, -8.07, 1e30, -10., 531.43, -5.]
    ffdata = [218.1727758412034, -7.068729635259748, 1e+30, -10.847020367291076, 530.508340450304, -4.879529128842498]

    return ffdata

