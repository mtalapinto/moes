import numpy as np
import copy

def to_next_surface(H, DC, z_1):
    
    Hout = np.zeros([len(H), 3])
    z0 = copy.copy(H[:, 2])
    z1 = np.full(len(H), z_1)
    d = z1 - z0
    # coordinates in the next surface paraxial plane
    Hout[:, 0] = (H[:, 0]+((DC[:, 0]/DC[:, 2])*d))
    Hout[:, 1] = (H[:, 1]+((DC[:, 1]/DC[:, 2])*d))
    Hout[:, 2] = H[:, 2] + d
    return Hout
