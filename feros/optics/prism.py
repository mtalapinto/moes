from . import refraction_index
import numpy as np
from . import transform
from . import trace
from . import refraction_index


def tracing(H, DC, Tin, l0, material, apex, decx, decy):
    DC_out = np.zeros([len(DC), 3])
    H_out = np.zeros([len(H), 3])

    H[:,2] = H[:,2] - H[:,2]
    #print(H)
    #print(DC)

    # Coordinate break prism in
    # Decenter
    H[:, 0] = H[:, 0] - decx
    H[:, 1] = H[:, 1] - decy
    # Tilt
    DC = -DC
    DC = transform.transform2(DC, -Tin)
    H = transform.transform2(H, -Tin)

    H_out[:, 0] = H[:, 0] - (DC[:, 0] / DC[:, 2]) * (H[:, 2])
    H_out[:, 1] = H[:, 1] - (DC[:, 1] / DC[:, 2]) * (H[:, 2])
    H_out[:, 2] = 0.

    #
    #tilt_y_sf_in = 32.2
    #T_sf_in = np.array([0, tilt_y_sf_in * np.pi / 180, 0])
    #DC = transform.transform(DC, -T_sf_in)
    #H_out = transform.transform(H_out, -T_sf_in)

    #H_out[:, 0] = H_out[:, 0] - (DC[:, 0] / DC[:, 2]) * (H_out[:, 2])
    #H_out[:, 1] = H_out[:, 1] - (DC[:, 1] / DC[:, 2]) * (H_out[:, 2])
    #H_out[:, 2] = 0.
    # End coordinate break

    # Prism 1 input surface refraction
    t = 20
    p = 1
    #n0 = refraction_index.nair_abs(l0, t, p)
    n0 = np.full(len(DC), 1)
    n1 = refraction_index.n(l0, t, p, material)
    n2 = np.full(len(DC), 1)
    #n2 = refraction_index.nair_abs(l0, t, p)
    n = np.zeros([len(DC), 3])
    n[:, 2] = 1.  # normal vector

    k = n1 / n0
    cosi = DC[:, 0] * n[:, 0] + DC[:, 1] * n[:, 1] + DC[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)

    DC_out[:, 0] = DC[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    prism_thic = np.full(len(H), -100.)
    H_out = trace.to_next_surface(H_out, DC_out, prism_thic)
    H_out[:, 2] = 0.
    # end prism 1 input surface

    # Coordinate break output surface
    decx2 = 52.
    decy2 = 0.0
    H_out[:, 0] = H_out[:, 0] - decx2
    H_out[:, 1] = H_out[:, 1] - decy2

    # Tilt
    T_sf_out = np.array([0, apex * np.pi / 180, 0])
    DC_out = transform.transform(DC_out, T_sf_out)
    H_out = transform.transform(H_out, T_sf_out)

    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.

    # end coordinate break output prism 1, alles gut bis hier

    # prism output surface
    k = n2 / n1
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)

    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]

    # End prism

    return H_out, DC_out