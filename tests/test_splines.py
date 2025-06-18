# bnpc/tests/test.py
import numpy as np
from npc.logSplines.utils import panelty_mat


def test_panelty_mat_non_singular():
    """
    Test that the penalty matrix is not singular for typical parameters.
    """
    # Example usage:
    knots = np.array([0, 0.25, 0.5, 0.75])
    d = 2  # order of difference
    degree = 3
    p_mat = panelty_mat(d=d, knots=knots, degree=degree, linear=False)
    assert p_mat.shape[0] == p_mat.shape[1], "Penalty matrix must be square."
    det_p_mat = np.linalg.det(p_mat)
    assert not np.isclose(
        det_p_mat, 0.0
    ), "Penalty matrix is singular or nearly singular."


def test_panelty_mat_linear():
    """
    Test that linear penalty matrix is computed as expected.
    """
    knots = np.array(
        [0, 1, 2]
    )
    d = 1
    p_mat = panelty_mat(d=d, knots=knots, degree=3, linear=True, k=5)
    assert p_mat.shape == (5, 5)
