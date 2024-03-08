from . import transform
import numpy as np


# [X, Y, Z]: [Echelle dispersion direction, cross-dispersion direction, optical axis direction]

def diffraction(H, DC, T, m, l, G):

    DC_out = np.zeros([len(DC), 3])
    Hout = np.zeros([len(DC), 3])

    DC = transform.transform2(DC, T)
    H = transform.transform2(H, T)

    Hout[:, 0] = H[:, 0] - (DC[:, 0] / DC[:, 2]) * (H[:, 2])
    Hout[:, 1] = H[:, 1] - (DC[:, 1] / DC[:, 2]) * (H[:, 2])
    Hout[:, 2] = 0.
    #DC[:, 2] = -DC[:, 2]

    if DC[0][1] == 0:
        signy = np.full(len(DC), 1)
    else:
        signy = (DC[:, 1]/np.abs(DC[:, 1]))

    signz = (DC[:, 2]/np.abs(DC[:, 2]))

    DC_out[:, 0] = -DC[:, 0]  # Reflection at the grating plane
    DC_out[:, 1] = signy*(m*l*G - np.abs(DC[:, 1]))
    DC_out[:, 2] = signz*(np.sqrt(1 - DC_out[:, 0]**2 - DC_out[:, 1]**2))
    DC_out[:, 2] = -DC_out[:, 2]  # Reflection at the grating plane

    DC_out_final = transform.transform2(DC_out, T)
    Hout = transform.transform2(Hout, T)

    Hout[:, 0] = Hout[:, 0] - (DC_out_final[:, 0] / DC_out_final[:, 2]) * (Hout[:, 2])
    Hout[:, 1] = Hout[:, 1] - (DC_out_final[:, 1] / DC_out_final[:, 2]) * (Hout[:, 2])
    Hout[:, 2] = 0.

    return Hout, DC_out_final
