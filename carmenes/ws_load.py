import numpy as np
import pandas as pd
from optics import parameters
from optics import env_data
from optics import vis_spectrometer


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

    #spectrum = np.array([data['order'].values, data['wlc'].values*1e-4])
    #spectrum = spectrum.T
    spectrum = pd.DataFrame()
    spectrum['order'] = data['order'].values
    spectrum['wave'] = data['wlc'].values * 1e-4
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

    rate = 0.05
    data = data.loc[data['posm'] >= 0]
    data = data.loc[data['posm'] <= 4250]
    data = data.loc[data['posmy'] >= 0]
    data = data.loc[data['posmy'] <= 4200]
    #data = data.loc[data['posc'] != 0.]
    data = data.loc[data['flag'] == 0]
    #data = data.loc[data['posme'] < 0.05]
    #
    #data = data.loc[np.abs(data['posm'] - data['posc']) < rate]
    data = data.dropna()
    print(len(data))
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


def load_ws_for_fit(date):
    path_data = 'data/vis_ws_timeseries/'
    #path_data = '/data/mtala/VIS_ws/'
    ws_data_A = pd.read_csv(path_data + str(date) + '/ws_hcl_A.csv', sep=',')
    ws_data_B = pd.read_csv(path_data + str(date) + '/ws_hcl_B.csv', sep=',')
    # Data filtering
    # No 0.0 values in computed position

    rate = 0.1
    ws_data_A = ws_data_A.loc[ws_data_A['posc'] != 0.00000]
    ws_data_A = ws_data_A.loc[ws_data_A['flag'] == 0]
    ws_data_A = ws_data_A.loc[np.abs(ws_data_A['posc'] - ws_data_A['posm']) <= rate]
    ws_data_A = ws_data_A.loc[ws_data_A['posme'] < rate]
    #ws_data_A = ws_data_A.loc[ws_data_A['posm'] < 0.19]
    #ws_data_A = ws_data_A.loc[ws_data_A['posm'] > -0.42]

    ws_data_B = ws_data_B.loc[ws_data_B['posc'] != 0.00000]
    ws_data_B = ws_data_B.loc[ws_data_B['flag'] == 0]
    ws_data_B = ws_data_B.loc[np.abs(ws_data_B['posc'] - ws_data_B['posm']) <= rate]
    ws_data_B = ws_data_B.loc[ws_data_B['posme'] < rate]

    '''
    wsa_data = ws_data_A.copy()
    wsb_data = ws_data_B.copy()
    print(wsa_data)
    #ws_data_A = ws_data_A[np.abs(ws_data_A.posm - ws_data_A.posc) < 0.05]
    #
    #ws_data_B = ws_data_B[np.abs(ws_data_B.posm - ws_data_B.posc) < 0.05]
    #wsa_data = np.array(wsa_data)
    #wsb_data = np.array(wsb_data)
    spec_a = spectrum_from_data(wsa_data)
    spec_b = spectrum_from_data(wsb_data)

    pressure = env_data.get_P_at_ws(date)
    init_state_a = parameters.load_date('A', date)
    init_state_b = parameters.load_date('B', date)
    init_state_a[-1] = pressure
    init_state_b[-1] = pressure
    temps = env_data.get_T_at_ws(date)
    wsa_model = vis_spectrometer.tracing(spec_a, init_state_a, 'A', temps)
    wsb_model = vis_spectrometer.tracing(spec_b, init_state_b, 'B', temps)
    
    
    resa = pd.DataFrame()
    resa['wlc'] = ws_data_A['wlc'].values
    resa['wll'] = ws_data_A['wll'].values
    resa['posc'] = ws_data_A['posc'].values
    resa['posm'] = ws_data_A['posm'].values
    resa['posme'] = ws_data_A['posme'].values
    resa['posmy'] = ws_data_A['posmy'].values
    resa['order'] = wsa_model['order']
    resa['flag'] = ws_data_A['flag'].values
    resa['dx'] = wsa_data['posm'] - wsa_model['x']
    resa['x'] = wsa_model['x']
    resa = resa.loc[resa['dx'] < 0.2]
    resa = resa.loc[resa['dx'] > -0.4]

    resb = pd.DataFrame()
    resb['wlc'] = ws_data_B['wlc'].values
    resb['wll'] = ws_data_B['wll'].values
    resb['posc'] = ws_data_B['posc'].values
    resb['posm'] = ws_data_B['posm'].values
    resb['posme'] = ws_data_B['posme'].values
    resb['posmy'] = ws_data_B['posmy'].values
    resb['order'] = ws_data_B['order'].values
    resb['flag'] = ws_data_B['flag'].values
    resb['dx'] = wsb_data['posm'] - wsb_model['x']
    resb['x'] = wsb_model['x']
    resb = resb.loc[resb['dx'] < 0.3]
    resb = resb.loc[resb['dx'] > -0.4]
    #plt.plot(resa['x'], resa['dx'], 'k.')
    #plt.show()
    resa = resa.drop(columns=['x', 'dx'])
    resb = resb.drop(columns=['x', 'dx'])
    #print(resa)
    print('Number of lines fiber A = ', len(resa))
    print('Number of lines fiber B = ', len(resb))
    print('\n')
    #print(resb)
    '''
    return ws_data_A, ws_data_B



if __name__ == '__main__':
    date = '2017-11-09'
    fib = 'A'
    kin = 'hcl'
    data = read_ws(date, kin, fib)
    print(data)
    import matplotlib.pyplot as plt
    plt.plot(data['posm'], data['posmy'], 'k.')
    plt.show()

