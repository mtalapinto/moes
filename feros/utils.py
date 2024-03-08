import numpy as np
import pandas as pd
from scipy.stats import norm, beta


def transform_uniform(x,a,b):
    return a + (b-a)*x


def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))


def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)


def transform_beta(x,a,b):
    return beta.ppf(x,a,b)


def fcam_vs_samp():
    fcam = np.arange(220, 400, 20)
    fcol = np.full(len(fcam), 876)
    slit = 100  # mum
    M = fcam/fcol
    slit_image = slit*M
    pixsize = 13.5  # mum
    sampling = slit_image/pixsize
    sampdata = pd.DataFrame()
    sampdata['fcam'] = fcam
    sampdata['sampling'] = sampling
    print(sampdata)
    return sampdata


if __name__ == '__main__':

    fcam_vs_samp()
