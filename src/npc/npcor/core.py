import numpy as np
from scipy.stats import gamma

"""
This file contains the core function for the non-parametric correction to a parametric noise model.
"""


#psd functions:
def dens(lam: np.ndarray, splines: np.ndarray) -> np.ndarray:
    """
    This function is defined to calculate the log of the density of the lambda
    :param lam: lambda
    :param splines: Splines
    :return: log density
    """
    return np.sum(lam[:, None] * splines, axis=0)



def psd(
    Snpar: np.ndarray, Spar: np.ndarray ) -> np.ndarray:

    S = Snpar + np.log(Spar)
    return S

def log_noise_psd(lam, splines, Spar):
    splines_psd = dens(
        lam=lam,
        splines=splines,
    )
    npsd = psd(
        Snpar=splines_psd,
        Spar=Spar,
    )
    if any(np.isnan(splines_psd)):
        raise ValueError("log spline PSD is nan")
    if any(np.isnan(npsd)):
        raise ValueError("log noise PSD is nan")

    return splines_psd, npsd



#priors:


def lamb_lprior(lam: np.ndarray, phi: float, P: np.ndarray, k: int) -> float:
    res=k * np.log(phi) / 2 - phi * np.matmul(np.transpose(lam), np.matmul(P, lam)) / 2
    if np.isnan(res):
        raise ValueError("lambda log prior is nan")
    return res


def phi_lprior(phi: float, delta: float) -> float:
    res=gamma.logpdf(phi, a=1, scale=1 / delta)
    if np.isnan(res):
        raise ValueError("phi log prior is nan")
    return res


def delta_lprior(delta: float) -> float:
    res=gamma.logpdf(delta, a=1e-4, scale=1 / 1e-4)
    if np.isnan(res):
        raise ValueError("delta log prior is nan")
    return res


def prior_sum(lam, phi, delta, P, k):
    '''
    This function calls the prior sum of splines
    :param lam: lambda
    :param phi: phi
    :param delta: delta
    :param P: Panelty matrix
    :param k: number of weights
    :return: prior sum of splines
    '''
    return lamb_lprior(
            lam=lam,
            phi=phi,
            P=P,
            k=k,
        )+ phi_lprior(
            phi=phi,
            delta=delta,
        )+ delta_lprior(
            delta=delta
        )




#loglikelihood and logposterior
def loglike(pdgrm, logpsd):
    lnlike = -1 * np.sum(logpsd + np.exp(np.log(pdgrm) - logpsd))
    return lnlike


def llike_cal(obj, npsdT):
    lnlike = loglike(pdgrm=obj.per, logpsd=npsdT[obj.mask])
    if np.isnan(lnlike):
        raise ValueError("log likelihood is nan")
    return lnlike

def logpost(loglike, logpri):
    return loglike + logpri


def post_calc(obj,lam,i):
    #function to calculate the posterior
    sp_psd_T, npsdT = log_noise_psd(lam=lam, splines=obj.splineobj.splines.T, Spar=obj.Spar)
    loglikelihood = llike_cal(obj, npsdT=npsdT)
    logpriorsum=prior_sum(lam=lam, phi=obj.splineobj.phi[i], delta=obj.splineobj.delta[i], P=obj.splineobj.P, k=obj.n_weights)
    lposterior=logpost(loglike=loglikelihood, logpri=logpriorsum)
    return sp_psd_T,npsdT,loglikelihood,logpriorsum,lposterior

