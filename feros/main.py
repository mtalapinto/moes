# This is a sample Python script.
import numpy as np
import matplotlib.pyplot as plt
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from optics import spectrograph
from optics import echelle_orders


def moes_zmx(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, moes for '+name)  # Press Ctrl+F8 to toggle the breakpoint.
    print('Performing ray tracing\n')
    spectrum = echelle_orders.init()
    wsmoes = spectrograph.tracing_paraxial(spectrum)


def moes_test():
    spectrum = echelle_orders.init()
    det = [2.02, 15, 4096, 2048, 0]

    wsmoes = spectrograph.tracing_det(spectrum, det)
    print(wsmoes['x'])
    print(wsmoes['wave'])
    plt.figure(figsize=(10, 5))  # fig size same as before
    ax = plt.gca()  # you first need to get the axis handle
    ax.set_aspect(1)
    plt.plot(wsmoes['x'], wsmoes['y'], 'k.')
    #plt.xlim(0, 4096)
    #plt.ylim(0, 2048)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #moes('feros')
    moes_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
