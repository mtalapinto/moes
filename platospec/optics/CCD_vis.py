import numpy as np
import matplotlib.pyplot as plt

def load():
    detector_size = 4096
    pix_size = 15e-3
    return detector_size, pix_size


def mm2pix(ws):
    pix_size_arr = np.full(len(ws), 15.e-3)
    detector_size_x = 4250.
    detector_size_y = 4200.
    ws[:, 2] = -ws[:, 2] 
    ws_ccd_y = ws[:, 3] / pix_size_arr + detector_size_x / 2
    ws_ccd_x = ws[:, 2] / pix_size_arr + detector_size_y / 2
    return ws_ccd_x, ws_ccd_y


def pix2mm_model(ws, coord):

    ws_out = np.copy(ws)
    pix_size_arr = np.full(len(ws), 15.e-3)
    detector_size_x = 4250.
    detector_size_y = 4200.

    detector_diagonal = np.sqrt(detector_size_x**2 + detector_size_y**2)
    if coord == 'x':
        norm = detector_diagonal*pix_size_arr
    else:
        norm = detector_diagonal*pix_size_arr

    ws_out[:, 3] = (ws[:, 3] - detector_size_x/2)*pix_size_arr/(norm/2)
    ws_out[:, 2] = (ws[:, 2] - detector_size_y/2)*pix_size_arr/(norm/2)

    return ws_out


def pix2mm_data(ws, coord):

    ws_out = np.copy(ws)
    pix_size_arr = np.full(len(ws), 15.e-3)
    detector_size_x = 4250.
    detector_size_y = 4200.

    if coord == 'x':
        norm = detector_size_x*pix_size_arr
    else:
        norm = detector_size_y*pix_size_arr

    ws_out[:, 5] = (ws[:, 5] - detector_size_x/2)*pix_size_arr/(norm/2)
    ws_out[:, 3] = (ws[:, 3] - detector_size_y/2)*pix_size_arr/(norm/2)

    return ws_out


def mm2pix_data_denorm(ws):

    pix_size_arr = np.full(len(ws), 15.e-3)
    detector_size_x = 4250.
    detector_size_y = 4200.



