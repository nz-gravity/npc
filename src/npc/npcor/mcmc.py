import numpy as np

from .sampler import MCMCResult, Sampler
from .input_test import input_test

"""
This file contains the function that runs the MCMC.
"""


def mcmc(
        per: np.ndarray,
        n: int,
        n_knots: int,
        burnin: int,
        Spar: np.ndarray = 1,
        degree: int = 3,
        f: np.ndarray = None,
        fs: float = None,
        blocked: bool = False,
        data_bin_edges: np.ndarray = None,
        data_bin_weights: np.ndarray = None,
        log_data: bool = False,
        equidistant: bool = False,
        thin: int = 1,
        amh: bool = False,
        cov_mat=np.nan,
        mean_w=None,
        num_iter_amh=1,
):
    """
    Function that:
      1) Validates the inputs,
      2) Initiates the sampler class,
      3) Runs MCMCloop(),
      4) Returns the results.
    """
    input_test(
        per=per, n=n, n_knots=n_knots, burnin=burnin, Spar=Spar, degree=degree,
        f=f, fs=fs, blocked=blocked, data_bin_edges=data_bin_edges,
        data_bin_weights=data_bin_weights, log_data=log_data,
        equidistant=equidistant, thin=thin, amh=amh,
    )
    sampler = Sampler(
        per=per,
        n=n,
        n_knots=n_knots,
        burnin=burnin,
        Spar=Spar,
        degree=degree,
        f=f,
        fs=fs,
        blocked=blocked,
        data_bin_edges=data_bin_edges,
        data_bin_weights=data_bin_weights,
        log_data=log_data,
        equidistant=equidistant,
        thin=thin,
        amh=amh,
        cov_mat=cov_mat,
        mean_w=mean_w,
        num_iter_amh=num_iter_amh,
    )
    sampler.MCMCloop()
    return MCMCResult(sampler=sampler)
