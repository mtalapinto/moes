import numpy as np
import copy
from decimal import *
getcontext().prec = 20

def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})


def to_next_surface(H, DC, z_1):
    set_numpy_decimal_places(20, 0)
    #Hout = np.zeros([len(H), 3])
    z0 = H[:, 2]
    z1 = np.full(len(H), z_1)
    d = z1 - z0
    # coordinates in the next surface paraxial plane
    H[:, 0] = (H[:, 0]+((DC[:, 0]/DC[:, 2])*d))
    H[:, 1] = (H[:, 1]+((DC[:, 1]/DC[:, 2])*d))
    H[:, 2] = H[:, 2] + d
    #print(H)
    return H
