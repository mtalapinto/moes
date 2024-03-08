import numpy as np
import matplotlib.pyplot as plt

def load():
    detector_size = 4096
    pix_size = 15e-3
    return detector_size, pix_size


def mm2pix(ws):
    pix_size_arr = np.full(len(ws), 15.e-3)
    detector_size_x = 2048.
    detector_size_y = 2048.
    ws['x'] = -ws['x'] 
    ws['y'] = ws['y'] / pix_size_arr + detector_size_x / 2
    ws['x'] = ws['x'] / pix_size_arr + detector_size_y / 2
    return ws


def mm2pix_custom(ws, det):
    pix_size_arr = np.full(len(ws), det[1]*1e-3)
    detector_size_x = det[2]
    detector_size_y = det[3]
    #ws['x'] = -ws['x']
    ws['y'] = ws['y'] / pix_size_arr + detector_size_x / 2
    ws['x'] = ws['x'] / pix_size_arr + detector_size_y / 2
    return ws
