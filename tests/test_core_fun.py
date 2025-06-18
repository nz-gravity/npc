import unittest
from math import isclose
import numpy as np

from npc.npcor.core import (
    delta_lprior,
    lamb_lprior,
    loglike,
    logpost,
    phi_lprior,
)


class TestCoreFunctions(unittest.TestCase):
    def test_lpost(self):
        """
        Test the log posterior combining log-likelihood and log-prior.
        """
        self.assertEqual(logpost(1.0, 2.0), 3.0)


    def test_loglike(self):
        """
        Test the loglike function, which calculates
        -sum( S + exp(log(pdgrm) - S) ).
        """
        pdgrm = np.array([10.0, 20.0])
        S = np.array([2.0, 3.0])
        expected = -7.349092  # approximate

        lnlike_val = loglike(pdgrm, S)
        self.assertAlmostEqual(lnlike_val, expected, places=5)

    def test_lamb_lprior(self):
        """
        Test lamb_lprior.
        This function is k*(log(phi))/2 - (phi/2)*(lam^T * P * lam).
        """
        lam = np.array([1.0, 2.0])
        phi = 0.5
        P = np.array([[2.0, 0.0], [0.0, 2.0]])  # a simple penalty matrix
        k = len(lam)  # = 2
        expected = -3.193147

        val = lamb_lprior(lam, phi, P, k)
        self.assertAlmostEqual(val, expected, places=5)

    def test_phi_lprior(self):
        """
        phi ~ Gamma(a=1, scale=1/delta).
        """
        val = phi_lprior(0.5, 1.0)
        self.assertTrue(isclose(val, -0.5, abs_tol=1e-5))

    def test_delta_lprior(self):
        """
        delta ~ Gamma(a=1e-4, scale=1 / 1e-4).
        """
        val = delta_lprior(1.0)
        self.assertFalse(np.isnan(val))



if __name__ == "__main__":
    unittest.main()
