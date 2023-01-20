import numpy as np
import pandas as pd


def carmenes_vis_ws():
    path_data = 'data/ws/'
    ws_data_A = pd.read_csv(path_data + '/ws_hcl_A.csv', sep=',')
    ws_data_B = pd.read_csv(path_data + '/ws_hcl_B.csv', sep=',')

    ws_data_A = ws_data_A.loc[ws_data_A['posc'] != 0.00000]
    ws_data_A = ws_data_A.loc[ws_data_A['posme'] < 0.75]

    ws_data_B = ws_data_B.loc[ws_data_B['posc'] != 0.00000]
    ws_data_B = ws_data_B.loc[ws_data_B['posme'] < 0.75]

    ws_data_A = ws_data_A[np.abs(ws_data_A.posm - ws_data_A.posc) < 0.1]
    ws_data_B = ws_data_B[np.abs(ws_data_B.posm - ws_data_B.posc) < 0.1]
    return ws_data_A, ws_data_B


def spectrum_from_ws(ws):
    ws = np.array(ws)
    spectrum = np.array([ws[:, -2], ws[:, 1]*1e-4])
    spectrum = spectrum.T
    return spectrum