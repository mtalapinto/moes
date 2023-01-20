import numpy as np
from . import transform


def flat_out(H, DC, T):
    
    DC_out = np.zeros([len(DC), 3])
    H_out = np.zeros([len(DC), 3])

    DC = transform.transform(DC, -T)
    H_out = transform.transform(H, -T)
    H_out[:, 0] = H[:, 0] - (DC[:, 0] / DC[:, 2]) * (H[:, 2])
    H_out[:, 1] = H[:, 1] - (DC[:, 1] / DC[:, 2]) * (H[:, 2])
    H_out[:, 2] = 0.

    # mirror default normal
    n0 = np.zeros([len(DC), 3])
    n0[:, 2] = 1
    n = transform.transform(n0, T)

    cosi = DC[:, 0] * n[:, 0] + DC[:, 1] * n[:, 1] + DC[:, 2] * n[:, 2]

    DC_out[:, 0] = DC[:, 0] - 2 * cosi * n[:, 0]
    DC_out[:, 1] = DC[:, 1] - 2 * cosi * n[:, 1]
    DC_out[:, 2] = DC[:, 2] - 2 * cosi * n[:, 2]

    DC_out = transform.transform2(DC_out, T)
    H_out = transform.transform2(H_out, T)
    H_out[:, 0] = H[:, 0] - (DC_out[:, 0] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 1] = H[:, 1] - (DC_out[:, 1] / DC_out[:, 2]) * (H_out[:, 2])
    H_out[:, 2] = 0.

    return H_out, DC_out
