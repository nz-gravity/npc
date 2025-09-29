import numpy as np
from collections import namedtuple
from scipy.stats import median_abs_deviation

"""
This file contains the utility functions used in the MCMC.
"""

def determinant(matrix, eps=1e-16):
    det_val = np.linalg.det(matrix)
    det_val = max(det_val, eps)
    return np.log(det_val)


def inverse(matrix):
    return np.linalg.pinv(matrix)

def basic_summaries(parameter):
    """
    Returns mean, standard deviation, and 95% credible interval for a parameter array.
    """
    mean_val = np.mean(parameter)
    std_val = np.std(parameter)
    ci_lower, ci_upper = np.percentile(parameter, [2.5, 97.5])
    return mean_val, std_val, ci_lower, ci_upper


def mad(x):
    med_abs_dav = np.median(abs(x - np.median(x)))
    if med_abs_dav == 0:
        med_abs_dav = 1e-10
    return med_abs_dav


def uniformmax(sample):
    median = np.median(sample)
    mad1 = mad(sample)
    abs_deviation = np.abs(sample - median)

    normalized_deviation = abs_deviation / mad1
    max_deviation = np.nanmax(normalized_deviation)

    return max_deviation


def cent_series(series):
    return ((series - np.mean(series)) / np.std(series))

def _absdiff(a,b):
    return abs(a-b)

def compute_iae(psd, truepsd, upper, lower):  # note use PSD not log PSD
    return sum(_absdiff(psd,truepsd)) * (upper-lower) / len(truepsd)

def compute_rel_iae(psd, truepsd, upper, lower):  # note use PSD not log PSD
    return sum(_absdiff(psd,truepsd)/truepsd) * (upper-lower) / len(truepsd)

def compute_prop(u05, u95, truepsd):
    v = []
    for x in range(len(u05)):
        if (truepsd[x] >= u05[x]) and (truepsd[x] <= u95[x]):
            v.append(1)
        else:
            v.append(0)
    return (np.mean(v))


def compute_ci(psds):
    CI = namedtuple('CI', ['u05', 'u95', 'med', 'label'])
    psd_help = np.apply_along_axis(uniformmax, 0, psds)
    psd_mad = median_abs_deviation(psds, axis=0)
    c_value = np.quantile(psd_help, 0.9)
    psd_med = np.median(psds, axis=0)
    psd_u95 = psd_med + c_value * psd_mad
    psd_u05 = psd_med - c_value * psd_mad
    return CI(u05=psd_u05, u95=psd_u95, med=psd_med, label='pypsd')


def ar1(n, a, sig):
    #AR1 time series
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = a * y[t-1] + np.random.normal(loc=0, scale=sig)
    return y

def ar2(n, a1, a2, sig):
    #AR2 time series
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = a1 * y[t-1] + a2 * y[t-2] + np.random.normal(loc=0, scale=sig)
    return y


def ar4(n, a1, a2, a3, a4, sig):
    #AR4 time series
    y = np.zeros(n)
    for t in range(4, n):
        y[t] = a1 * y[t-1] + a2 * y[t-2] + a3 * y[t-3] + a4 * y[t-4] + np.random.normal(loc=0, scale=sig)
    return y


def s_ar1(a, sig2, f):
    #AR1 spectrum
    return sig2 / (1 + a**2 - 2 * a * np.cos(2 * np.pi * f))


def s_ar4(a1, a2, a3, a4, sig2, f):
    #AR(4) spectrum
    psd=sig2/(np.abs(1-a1*np.exp(1j*2*np.pi*f)-a2*np.exp(1j*4*np.pi*f)-a3*np.exp(1j*6*np.pi*f)-a4*np.exp(1j*8*np.pi*f)))**2
    return psd

def s_ar2(a1, a2, sig2, f):
    #AR2 spectrum
    return sig2 / (1 + a1**2 + a2**2 + 2 * a1 * (a2 - 1) * np.cos(2 * np.pi * f) - 2 * a2 * np.cos(4 * np.pi * f))

def updateCov(X, covObj=np.nan):
    if not isinstance(covObj, dict):  # np.isnan(covObj):
        covObj = {'mean': X, 'cov': np.nan, 'n': 1}
        return covObj

    covObj['n'] = covObj['n'] + 1

    X1 = np.asmatrix(covObj['mean'])
    X = np.asmatrix(X)

    if covObj['n'] == 2:
        covObj['mean'] = X1 / 2 + X / 2
        dX1 = X1 - covObj['mean']
        dX2 = np.subtract(X, covObj['mean'])
        dX1 = np.asmatrix(dX1)
        dX2 = np.asmatrix(dX2)
        covObj['cov'] = np.add(np.matmul(dX1.transpose(), dX1), np.matmul(dX2.transpose(), dX2))
        return covObj

    dx = X1 - X  # previous mean minus new X
    dx = np.asmatrix(dx)
    covObj['cov'] = covObj['cov'] * (covObj['n'] - 2) / (covObj['n'] - 1) + \
                    np.matmul(dx.transpose(), dx) / covObj['n']
    covObj['mean'] = X1 * (covObj['n'] - 1) / covObj['n'] + X / covObj['n']
    return covObj
