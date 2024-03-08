from . import refraction_index
import numpy as np
from . import transform
from . import trace


def dispersion(H, DC, Tin, l0, material, apex, GD, t, p, dec_x, dec_y):
                                                                   
    DC_out = np.zeros([len(DC), 3])
    H_out = np.zeros([len(H), 3])

    # Coordinate break 0
    # Decenter
    H[:, 0] = H[:, 0] + dec_x
    H[:, 1] = H[:, 1] + dec_y
    #Tilt
    DC = transform.transform(DC, -Tin)
    H = transform.transform(H, -Tin)
    H_out[:, 0] = H[:, 0] - (DC[:, 0] / DC[:, 2]) * (H[:, 2])
    H_out[:, 1] = H[:, 1] - (DC[:, 1] / DC[:, 2]) * (H[:, 2])
    H_out[:, 2] = 0.
    # End coordinate break 0

    # Grism input surface
    tilt_y_sf_in = 6.
    T_sf_in = np.array([0, tilt_y_sf_in*np.pi/180, 0])
    DC = transform.transform(DC, -T_sf_in)

    H_out = transform.transform(H_out, -T_sf_in)
    H_out[:, 0] = H_out[:, 0] - (DC[:, 0] / DC[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC[:, 1] / DC[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.

    # Refractive indices
    n0 = refraction_index.nair_abs(l0, t, p)
    n1 = refraction_index.n(l0, t, p, material)
    n2 = refraction_index.nair_abs(l0, t, p)  # output in air

    # surface default normal
    n = np.zeros([len(DC), 3])
    n[:, 2] = -1.

    k = n1/n0
    cosi = DC[:, 0]*n[:, 0] + DC[:, 1]*n[:, 1] + DC[:, 2]*n[:, 2]
    sini = np.sqrt(1 - cosi**2)
    sinr = sini/k
    cosr = np.sqrt(1 - sinr**2)

    DC_out[:, 0] = DC[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC[:, 2] / k + (cosr - cosi / k) * n[:, 2]
    # End grism first surface, alles gut bei hier

    DC_out = transform.transform2(DC_out, T_sf_in)
    H_out = transform.transform2(H_out, T_sf_in)
    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.


    # Trace to next surface
    d_trf_col = np.full(len(H), -40.)
    H_out = trace.to_next_surface(H_out, DC_out, d_trf_col)
    H_out[:, 2] = 0.

    # Coordinate break 1
    T_sf_out = np.array([0, -(apex - 6)*np.pi/180, 0])
    DC_out = transform.transform(DC_out, -T_sf_out)

    H_out = transform.transform(H_out, -T_sf_out)
    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.

    # End coordinate break 1

    # Grism output surface
    # Snell law at aperture surface
    k = n2 / n1
    cosi = DC_out[:, 0] * n[:, 0] + DC_out[:, 1] * n[:, 1] + DC_out[:, 2] * n[:, 2]
    sini = np.sqrt(1 - cosi ** 2)
    sinr = sini / k
    cosr = np.sqrt(1 - sinr ** 2)

    DC_out[:, 0] = DC_out[:, 0] / k + (cosr - cosi / k) * n[:, 0]
    DC_out[:, 1] = DC_out[:, 1] / k + (cosr - cosi / k) * n[:, 1]
    DC_out[:, 2] = DC_out[:, 2] / k + (cosr - cosi / k) * n[:, 2]
    # End of Snell law at aperture, alles gut

    # Coordinate break 2
    T_sf_out_grating = np.array([0, 0, 90*np.pi/180])
    DC_out = transform.transform(DC_out, -T_sf_out_grating)
    H_out = transform.transform(H_out, -T_sf_out_grating)
    # End of coordinate break 2, alles gut

    # Diffraction grating
    signz = (DC[:, 2] / np.abs(DC[:, 2]))

    m = 1
    DC_out[:, 0] = DC_out[:, 0]
    DC_out[:, 1] = (m * l0 * GD + DC_out[:, 1])
    DC_out[:, 2] = signz * (np.sqrt(1 - DC_out[:, 0] ** 2 - DC_out[:, 1] ** 2))
    # End diffraction

    # Coordinate break 3
    DC_out = transform.transform(DC_out, T_sf_out_grating)
    DC_out = transform.transform(DC_out, T_sf_out)
    DC_out[:, 2] = signz * (np.sqrt(1 - DC_out[:, 0] ** 2 - DC_out[:, 1] ** 2))
    H_out = transform.transform(H_out, T_sf_out_grating)
    H_out = transform.transform(H_out, T_sf_out)
    H_out[:, 0] = H_out[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H_out[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.
    # End grism

    return H_out, DC_out
