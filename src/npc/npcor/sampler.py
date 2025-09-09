import numpy as np
from ..logSplines.logspline_data import LogSplineData
from ..logSplines.log_Psplines import logPsplines
from .core import post_calc
from ..logSplines.utils import update_phi_delta
from .update_weights import update_lambda

"""This file contains the Sampler class which is used to run the MCMC."""
class Sampler:
    def __init__(
            self,
            per: np.ndarray,
            n: int,
            n_weights: int,
            burnin: int,
            Spar: np.ndarray,
            degree: int,
            f: np.ndarray ,
            fs: float ,
            blocked: bool,
            data_bin_edges: np.ndarray,
            data_bin_weights: np.ndarray,
            log_data: bool,
            equidistant: bool,
            thin: int ,
            amh: bool ,
            cov_mat,
    ):
        self.per=np.mean(per, axis=0) if blocked else per
        self.J = per.shape[0] if blocked else 1
        self.n = n
        self.thin = thin
        self.n_weights = n_weights
        self.burnin = burnin
        self.Spar = Spar
        self.degree = degree
        self.fs = fs
        self.blocked = blocked
        self.data_bin_edges = data_bin_edges
        self.data_bin_weights = data_bin_weights
        self.log_data = log_data
        self.equidistant = equidistant
        self.amh=amh
        if f is None:
            f = np.linspace(0, self.fs / 2, len(self.per) + 1)[1:]
        self.f = f
        dataobj_x = LogSplineData(
            data=self.per,
            Spar=self.Spar,
            n=self.n,
            n_knots=self.n_weights,
            degree=self.degree,
            f=self.f,
            data_bin_edges=self.data_bin_edges,
            data_bin_weights=self.data_bin_weights,
            log_data=self.log_data,
            equidistant=self.equidistant,
        )
        self.splineobj = logPsplines(dataobj=dataobj_x)
        self.splineobj.a_phi = n_weights / 2 + 1
        self.splineobj.a_delta = 1 + 1e-4
        self.splineobj.count = []
        self.splineobj.n_gridpoints, self.splineobj.n_basis = self.splineobj.splines.shape
        self.splineobj.beta_cov=0.05
        self.splineobj.const=(2.38**2)/self.n_weights
        self.splineobj.covobj = {'mean': self.splineobj.lam_mat[0, :], 'cov': cov_mat, 'n': 1}
        self.splineobj.Ik = (0.1)**2*np.diag(np.ones(self.n_weights) / self.n_weights)
        #self.splineobj.covobj = covobj if covobj is not None else np.eye(self.n_weights)
        self.splineobj.sigma = 1
        self.splineobj.accept_frac = 0.4
        self.npsd = np.zeros((n, len(self.per)))  # noise PSD T channel
        self.loglikelihood = np.zeros(n)
        self.logpost = np.zeros(n)
        self.logpriorsum = np.zeros(n)
        bFreq = [0, len(self.per) - 1]  # Remove the first and last elements
        mask = np.ones(len(self.per), dtype=bool)
        mask[bFreq] = False
        self.mask=mask
        self.splineobj.splines_mat[0, :], self.npsd[0, :], self.loglikelihood[0], self.logpriorsum[0], self.logpost[
            0] = post_calc(
            self, lam=self.splineobj.lam_mat[0, :], i=0)

    def MCMCloop(self):
        for i in range(1, self.n):
            update_lambda(obj=self, i=i)
            update_phi_delta(self, ind=i)
            self.splineobj.splines_mat[i, :], self.npsd[i, :], self.loglikelihood[i], self.logpriorsum[i], self.logpost[
                i] = post_calc(self, lam=self.splineobj.lam_mat[i, :], i=i)


class MCMCResult:
    def __init__(self, sampler):
        """
        Take a completed Sampler instance (after MCMCloop() has run),
        slice out the burn-in region, and store the MCMC outputs.

        Parameters
        ----------
        sampler : Sampler
            A Sampler instance that has run MCMCloop().
        """
        # Burn-in region
        self.burnin = sampler.burnin
        self.n = sampler.n
        self.n_weights = sampler.n_weights
        self.blocked = sampler.blocked
        self.thin=sampler.thin
        cond = slice(self.burnin, self.n, self.thin)

        self.phi = sampler.splineobj.phi[cond]
        self.delta = sampler.splineobj.delta[cond]

        self.loglikelihood = sampler.loglikelihood[cond]
        self.logpriorsum = sampler.logpriorsum[cond]
        self.logpost = sampler.logpost[cond]

        # Noise PSD
        self.noise_psd = sampler.npsd[cond, :]
        self.splines_psd = sampler.splineobj.splines_mat[cond, :]  # spline PSD

        self.lambda_matrix = sampler.splineobj.lam_mat[cond, :]

        self.knots = sampler.splineobj.knots
        self.iter_ix = np.arange(self.burnin, self.n)[::self.thin]#the iterations after burn-in and thinning
        self.n_kept = len(self.iter_ix)#Number of posterior samples kept after burn-in and thinning
        self.amh = sampler.amh
        self.covobj = sampler.splineobj.covobj



