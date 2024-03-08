import pyzdde.zdde as pyz
import numpy as np
import echelle_orders
import matplotlib.pyplot as plt


def ray_tracing(sfno, spectra):
    hx = 0
    hy = 0
    #print(spectra)
    ln = pyz.createLink()
    s = []
    echelle_sf_no = 4  # Echelle par1: grating constant, par2: order number

    for i in range(len(spectra)):
        ln.zSetSurfaceParameter(echelle_sf_no, 2, int(spectra[i][0]))
        ln.zSetWave(1, float(spectra[i][1]), 1.)  # Set wavelength to trace
        #print(ln.zGetWave())
        ln.zGetUpdate()
        #nindex = ln.zGetIndex(sfno)
        rayTraceData = ln.zGetTrace(1, 0, int(sfno), hx, hy, 0, 0)  # function that performs the tracing; wave, mode, surf. (-1 for image plane)
        error, vig, x, y, z, l, m, n, l2, m2, n2, intensity = rayTraceData
        s.append([spectra[i][0], spectra[i][1], x, y, z, l, m, n])#, nindex])
        #print(spectra[i][0], spectra[i][1], x, y)

    pyz.closeLink(ln)
    return np.array(s)


def tracing_test():
    spectra = echelle_orders.init()
    print(spectra)


if __name__ == '__main__':
    sfno = -1
    spectrum = echelle_orders.init()
    s = ray_tracing(sfno, spectrum)
    print(s)
    #s = np.array(s)

    #tracing_test()
    plt.figure(figsize=[8, 4])
    plt.plot(s[:, 3], s[:, 2], '.')
    plt.show()