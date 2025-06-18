import numpy as np
from .core import post_calc

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

def update_lambda(obj,i):
    #function to update lambda
    obj.splineobj.sigma = sigmaupdate(
        accept_frac=obj.splineobj.accept_frac, sigma=obj.splineobj.sigma
    )
    (
        obj.splineobj.lam_mat[i, :],
        obj.splineobj.accept_frac,
    ) = lambda_loop(obj, i=i-1)
