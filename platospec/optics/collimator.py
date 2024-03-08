import numpy as np
from . import transform
#import compare_zemax


def DCcoll(H, DCs, T, r):

    n = np.zeros([len(H), 3])
    DC_out = np.zeros([len(DCs), 3])
    H_out = np.zeros([len(H), 3])

    old_z_coor = H[:, 2].copy()

    H[:, 2] = 0
    H_comp = H.copy()

    H = transform.transform(H, -T)
    DCs = transform.transform(DCs, -T)

    x_p = H[:, 0]
    y_p = H[:, 1]
    T_x = DCs[:, 0]/DCs[:, 2]
    T_y = DCs[:, 1]/DCs[:, 2]

    r1 = 2*r
    dz = (-(x_p*T_x + y_p*T_y - r1/2) - r1/np.abs(r1)*np.sqrt((r1/2)**2 - r1*(x_p*T_x + y_p*T_y) - (y_p*T_x - x_p*T_y)**2))/(T_x**2 + T_y**2)

    H_comp[:, 0] = H[:, 0] + dz * T_x
    H_comp[:, 1] = H[:, 1] + dz * T_y
    H_comp[:, 2] = dz

    # Coordinates at collimator surface
    H[:, 0] = H[:, 0] + dz*T_x
    H[:, 1] = H[:, 1] + dz*T_y
    H[:, 2] = old_z_coor + dz

    # Normal vector at collimator surface
    n[:, 0] = -H[:, 0]/np.sqrt(H[:, 0]**2 + H[:, 1]**2 + r**2)
    n[:, 1] = -H[:, 1]/np.sqrt(H[:, 0]**2 + H[:, 1]**2 + r**2)
    n[:, 2] = r/np.sqrt(H[:, 0]**2 + H[:, 1]**2 + r**2)

    cosi = DCs[:, 0] * n[:, 0] + DCs[:, 1] * n[:, 1] + DCs[:, 2] * n[:, 2]

    DC_out[:, 0] = DCs[:, 0] - 2 * cosi * n[:, 0]
    DC_out[:, 1] = DCs[:, 1] - 2 * cosi * n[:, 1]
    DC_out[:, 2] = DCs[:, 2] - 2 * cosi * n[:, 2]

    DC_out = transform.transform(DC_out, T)
    signz = DC_out[:, 2] / np.abs(DC_out[:, 2])
    DC_out[:, 2] = signz * np.sqrt(1 - DC_out[:, 0] ** 2 - DC_out[:, 1] ** 2)

    #file_comp = 'ws_zemax_coll3.txt'
    #compare_zemax.difference_z_negative(H_comp, DC_out, file_comp)
    return H, DC_out
