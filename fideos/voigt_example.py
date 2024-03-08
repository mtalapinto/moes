import numpy as np
from scipy.special import wofz
import pylab

def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def V(x, amp, mean, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return amp*np.real(wofz(((x - mean) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

alpha, gamma = 10.5, 10.5
amp = 1.e2
x = np.linspace(-10.,10.,1000)
mean = 1.
#pylab.plot(x, G(x, alpha), ls=':', label='Gaussian')
#pylab.plot(x, L(x, gamma), ls='--', label='Lorentzian')
pylab.plot(x, V(x, amp, mean, alpha, gamma), label='Voigt')
#pylab.xlim(-0.8,0.8)
pylab.legend()
pylab.show()
