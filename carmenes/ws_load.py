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


def spectrum_from_data(data):

    spectrum = np.array([data['order'].values, data['wlc'].values*1e-4])
    spectrum = spectrum.T
    return spectrum


def load_ws(date, kin, fib):
    path_data = 'data/vis_ws_timeseries/'
    if kin == 'fp':
        if fib == 'A':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv' )
        elif fib == 'B':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv')
        else:
            print('No fiber '+str(fib))
    elif kin == 'hcl':
        if fib == 'A':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv' )
        elif fib == 'B':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv')
        else:
            print('No fiber '+str(fib))
    else:
        print('Insert right option: fp or hcl')


    data = data.loc[data['posm'] >= 0]
    data = data.loc[data['posm'] <= 4250]
    data = data.loc[data['posmy'] >= 0]
    data = data.loc[data['posmy'] <= 4200]
    #data = data.loc[data['posc'] != 0.]
    #data = data.loc[data['flag'] == 0]
    #data = data.loc[data['posme'] < 1.]
    #
    #data = data.loc[np.abs(data['posm'] - data['posc']) < rate]
    data = data.dropna()
    return data


def read_ws(date, kin, fib):
    path_data = 'data/vis_ws_timeseries/'
    if kin == 'fp':
        if fib == 'A':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv' )
        elif fib == 'B':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv')
        else:
            print('No fiber '+str(fib))
    elif kin == 'hcl':
        if fib == 'A':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv' )
        elif fib == 'B':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv')
        else:
            print('No fiber '+str(fib))
    else:
        print('Insert right option: fp or hcl')


    data = data.loc[data['posm'] >= 0]
    data = data.loc[data['posm'] <= 4250]
    data = data.loc[data['posmy'] >= 0]
    data = data.loc[data['posmy'] <= 4200]
    data = data.loc[data['posc'] != 0.]
    data = data.loc[data['flag'] == 0]
    #data = data.loc[data['posme'] < 1.]
    #
    #data = data.loc[np.abs(data['posm'] - data['posc']) < rate]
    data = data.dropna()
    return data


def read_ws_from_spec(date, kin, fib, spectrum):
    path_data = 'data/vis_ws_timeseries/'
    if kin == 'fp':
        if fib == 'A':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv' )
        elif fib == 'B':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv')
        else:
            print('No fiber '+str(fib))
    elif kin == 'hcl':
        if fib == 'A':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv' )
        elif fib == 'B':
            data = pd.read_csv(path_data + date + '/ws_' + str(kin) + '_' + str(fib) + '.csv')
        else:
            print('No fiber '+str(fib))
    else:
        print('Insert right option: fp or hcl')

    data = data.loc[data['posm'] >= 0]
    data = data.loc[data['posm'] <= 4250]
    data = data.loc[data['posmy'] >= 0]
    data = data.loc[data['posmy'] <= 4200]
    data = data.loc[data['posc'] != 0.]
    data = data.loc[data['flag'] == 0]
    data = data.dropna()
    data['order'] = data['order'].astype(int)
    data['wll'] = data['wll'].round(6)
    dataout = pd.merge(data, spectrum, on=['order', 'wll'], how='inner')    #data = data.loc[data['posme'] < 1.]
    dataout = dataout.drop_duplicates(subset=['wll', 'order'])
    #
    #data = data.loc[np.abs(data['posm'] - data['posc']) < rate]
    return dataout



if __name__ == '__main__':
    date = '2017-11-09'
    fib = 'A'
    kin = 'hcl'
    data = read_ws(date, kin, fib)
    print(data)
    import matplotlib.pyplot as plt
    plt.plot(data['posm'], data['posmy'], 'k.')
    plt.show()

