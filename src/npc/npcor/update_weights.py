import numpy as np
from .core import post_calc
from scipy.stats import multivariate_normal
from npc.utils import updateCov
"""This file contains the functions to update the weights (lambda) in the MCMC sampler."""


def sigmaupdate(accept_frac: float, sigma: float) -> float:
    if accept_frac < 0.30:
        sigma *= 0.90
    elif accept_frac > 0.50:
        sigma *= 1.10
    return sigma

def lambda_loop(obj,i):
    #loop to update lambda
    lam = obj.splineobj.lam_mat[i, :] #assuming the i is the previous index
    accept_count = 0
    aux = np.arange(0, obj.n_weights)
    np.random.shuffle(aux)
    for sth in range(0, len(lam)):
        logu = np.log(np.random.rand())
        pos = aux[sth]
        z = np.random.normal()
        lam_p = lam[pos]

        _, _,_,_,ftheta = post_calc(obj,lam=lam,i=i)

        lam[pos] = lam_p + obj.splineobj.sigma * z#lambda star
        _, _,_,_,ftheta_star = post_calc(obj,lam=lam,i=i)

        fac = ftheta_star - ftheta

        if np.isnan(fac):
            fac = -1e9

        if logu <= fac:
            accept_count += 1
        else:
            # Reject update
            lam[pos] = lam_p

    accept_frac = accept_count / obj.n_weights
    return lam, accept_frac

def update_lambda_mh(obj,i):
    #function to update lambda
    obj.splineobj.sigma = sigmaupdate(
        accept_frac=obj.splineobj.accept_frac, sigma=obj.splineobj.sigma
    )
    (
        obj.splineobj.lam_mat[i, :],
        obj.splineobj.accept_frac,
    ) = lambda_loop(obj, i=i-1)

def get_lamstar(obj,lam,i):
        u=np.random.rand()
        if  (i <= 50*obj.n_weights) or (u<0.05):
             return multivariate_normal.rvs(mean=lam, cov=obj.splineobj.Ik , size=1)
        return multivariate_normal.rvs(mean=lam, cov=obj.splineobj.const*obj.splineobj.covobj['cov'], size=1)

def lambda_amh_loop(obj,i):
    #loop to update lambda using amh
    lam = obj.splineobj.lam_mat[i, :] #assuming the i is the previous index
    _, _, _, _, ftheta = post_calc(obj, lam=lam, i=i)
    lam_star=get_lamstar(obj,lam,i)
    _, _, _, _, ftheta_star = post_calc(obj, lam=lam_star, i=i)
    logu = np.log(np.random.rand())
    fac = ftheta_star - ftheta
    if np.isnan(fac):
        fac = -1e9
    if logu <= fac:
        return lam_star
    return lam


def update_lambda_amh(obj, i, epsilon=1e-11, adaptation_delay=100, adapt_every=10):
    #function to update lambda
    obj.splineobj.lam_mat[i, :]=lambda_amh_loop(obj, i=i-1)
    obj.splineobj.covobj=updateCov(X=obj.splineobj.lam_mat[i, :], covObj=obj.splineobj.covobj)
    #if (i > adaptation_delay) and (i % adapt_every == 0):
    #    samples_so_far = obj.splineobj.lam_mat[:i + 1, :]
    #    cov_emp = np.cov(samples_so_far, rowvar=False)
    #    obj.splineobj.covobj = cov_emp + epsilon * np.eye(obj.n_weights)



def update_lambda(obj,i):
    if obj.amh:
        update_lambda_amh(obj,i)
    else:
        update_lambda_mh(obj, i)
