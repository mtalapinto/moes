import pyzdde.zdde as pyz
import numpy as np
import echelle_orders
import matplotlib.pyplot as plt
from decimal import *
getcontext().prec = 8
np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

def ray_tracing(sfno):
    hx = 0
    hy = 0
    spectra = echelle_orders.init()
    ln = pyz.createLink()
    s = []

    for i in range(len(spectra)):
        ln.zSetSurfaceParameter(5, 2, int(spectra[i][0]))
        ln.zSetWave(1, float(spectra[i][1]), 1.)  # Set wavelength to trace
        ln.zGetUpdate()
        # indexData = ln.zGetIndex(sfno)
        rayTraceData = ln.zGetTrace(1, 0, sfno, hx, hy, 0, 0)  # function that performs the tracing; wave, mode, surf. (-1 for image plane)
        error, vig, x, y, z, l, m, n, l2, m2, n2, intensity = rayTraceData
        #s.append([spectra[i][0], spectra[i][1], x, y, z])
        s.append([x, y, z, l, m, n])

    pyz.closeLink(ln)
    return np.array(s)

def fideos_tracing(sfno):
    s = ray_tracing(sfno)
    return s

if __name__ == '__main__':
    sf = -1
    s = fideos_tracing(sf)
    #print(s)
    s = np.array(s)
    plt.plot(s[:,2], s[:,3],'.')
    plt.show()