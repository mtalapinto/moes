import numpy as np
from optics import trace
from decimal import *
getcontext().prec = 20


def tracing(H, DC, f, thic):
    power = 1 / f
    x_bar = DC[:, 0]/DC[:, 2] - H[:, 0] * power
    y_bar = DC[:, 1]/DC[:, 2] - H[:, 1] * power
    dc_z = 1/np.sqrt(x_bar**2 + y_bar**2 + 1)

    DC[:, 0] = dc_z*x_bar
    DC[:, 1] = dc_z*y_bar
    DC[:, 2] = np.sqrt(1. - DC[:, 0] ** 2 - DC[:, 1]**2)

    d_trf_col = np.full(len(H), thic)
    #print(H)
    H_out = trace.to_next_surface(H, DC, d_trf_col)
    H_out[:, 2] = 0.
    #print(H_out)
    return H_out, DC

