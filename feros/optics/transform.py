import numpy as np

# coordinates rotation function


def transform(DC, T):
    
    Rz = np.matrix(((np.cos(T[2]), np.sin(T[2]), 0), (-np.sin(T[2]), np.cos(T[2]), 0), (0, 0, 1)))
    Ry = np.matrix(((np.cos(T[1]), 0, np.sin(T[1])), (0, 1, 0), (-np.sin(T[1]), 0, np.cos(T[1]))))
    Rx = np.matrix(((1, 0, 0), (0, np.cos(T[0]), np.sin(T[0])), (0, -np.sin(T[0]), np.cos(T[0]))))
    
    DC_matrix = np.matrix((DC[:, 0], DC[:, 1], DC[:, 2]))
    DC_out = np.squeeze(np.asarray(Rx*Ry*Rz*DC_matrix)).transpose()
    return DC_out


def transform2(DC, T):
    Rz = np.matrix(((np.cos(T[2]), np.sin(T[2]), 0), (-np.sin(T[2]), np.cos(T[2]), 0), (0, 0, 1)))
    Ry = np.matrix(((np.cos(T[1]), 0, np.sin(T[1])), (0, 1, 0), (-np.sin(T[1]), 0, np.cos(T[1]))))
    Rx = np.matrix(((1, 0, 0), (0, np.cos(T[0]), -np.sin(T[0])), (0, np.sin(T[0]), np.cos(T[0]))))

    DC_matrix = np.matrix((DC[:, 0], DC[:, 1], DC[:, 2]))
    DC_out = np.squeeze(np.asarray(Rz * Ry * Rx * DC_matrix)).transpose()
    return DC_out
