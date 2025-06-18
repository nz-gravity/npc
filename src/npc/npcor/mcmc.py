import numpy as np

from .sampler import MCMCResult, Sampler

"""
This file contains the function that runs the MCMC.
"""


def mcmc(
        per: np.ndarray,
        n: int,
        n_weights: int,
        burnin: int,
        Spar: np.ndarray = 1,
        degree: int = 3,
        f: np.ndarray = None,
        fs: float = None,
        blocked: bool = False,
        data_bin_edges: np.ndarray = None,
        data_bin_weights: np.ndarray = None,
        log_data: bool = False,
        equidistant: bool = True,
):
    """
    Function that:
      1) Initiates the sampler class,
      2) Runs MCMCloop(),
      3) Returns the results.
    """
    sampler = Sampler(
        per=per,
        n=n,
        n_weights=n_weights,
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
    )
    sampler.MCMCloop()
    return MCMCResult(sampler=sampler)
