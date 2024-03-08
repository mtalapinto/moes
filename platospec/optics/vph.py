import numpy as np
from optics import transform
from optics import refraction_index
from optics import trace


def tracing(H, DC, l0, material, Tin):
    DC_out = np.zeros([len(DC), 3])
    H_out = np.zeros([len(H), 3])

    dec_x = -100.
    dec_y = 0.
    # Coordinate break 0
    # Decenter
    H[:, 0] = H[:, 0] - dec_x
    H[:, 1] = H[:, 1] - dec_y
    # Tilt
    DC = transform.transform(DC, -Tin)
    H = transform.transform(H, -Tin)
    H_out[:, 0] = H[:, 0] - (DC[:, 0] / DC[:, 2]) * (H[:, 2])
    H_out[:, 1] = H[:, 1] - (DC[:, 1] / DC[:, 2]) * (H[:, 2])
    H_out[:, 2] = 0.
    # End coordinate break 0

    # Refractive indices
    t = 20.
    p = 1.

    n0 = np.full(len(H), 1.)
    n1 = refraction_index.n(l0, t, p, material)
    n2 = np.full(len(H), 1.)
    #n2 = refraction_index.nair_abs(l0, t, p)  # output in air

    # surface default normal
    n = np.zeros([len(DC), 3])
    n[:, 2] = 1.

    k = n1 / n0
    cosi = DC[:, 0] * n[:, 0] + DC[:, 1] * n[:, 1] + DC[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)

    DC_out[:, 0] = DC[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC[:, 2] / k + (cosr - cosi / k) * n[:, 2]
    # End vph first surface

    d_sf01 = np.full(len(H), -8.)
    H_out = trace.to_next_surface(H_out, DC_out, d_sf01)
    H_out[:, 2] = 0.

    # VPH output surface
    k = n2 / n1
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)

    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # Diffraction grating
    signz = (DC_out[:, 2] / np.abs(DC_out[:, 2]))
    GD = 340 * 1e-3
    m = -1
    DC_out[:, 0] = DC_out[:, 0]
    DC_out[:, 1] = (m * l0 * GD + DC_out[:, 1])
    DC_out[:, 2] = signz * (np.sqrt(1 - DC_out[:, 0] ** 2 - DC_out[:, 1] ** 2))
    # End diffraction

    DC_out = transform.transform(DC_out, Tin)
    H_out = transform.transform(H_out, Tin)

    return H_out, DC_out



