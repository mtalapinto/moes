from scipy.stats import norm,beta
import numpy as np


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
