import echelle_orders
import numpy as np


def do_slit_files(rv, fcam):
    echelle_orders.init_stellar_doppler(rv, fcam)
    print('spectrum file written... rv = ', rv, 'fcam = ', fcam, '\n')


def do_slit_files_all(fcam):
    rvs = np.arange(-6500, 10001, 250)
    for rv in rvs:
        do_slit_files(rv, fcam)


def do_slit_order(rv, fcam, omin):
    echelle_orders.init_stellar_doppler(rv, fcam)
    print('spectrum file written... rv = ', rv, 'fcam = ', fcam, '\n')


if __name__ == '__main__':

    do_slit_files(0, 300)
    #do_slit_files_all(230)
    import time
    time.sleep(10)