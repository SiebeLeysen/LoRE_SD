import numpy as np

from utils import SphericalHarmonics as sh

def non_negative_odf(Q):
    """
    Non-negative ODF constraint. ODF must be non-negative along the directions defined in Q (typically 300).
    :param Q: Transformation matrix between angular domain and SH
    :return: dictionary according to scipy.optimize's NonLinearConstraint
    """
    return {'type': 'ineq', 'fun': lambda x: Q @ x[:Q.shape[-1]]}

def sum_of_fractions_equals_one(lmax):
    """
    All fractions used in the respones functions representation must sum to unity.
    :param lmax: Max SH order
    :return: dictionary according to scipy.optimize's NonLinearConstraint
    """
    return {'type': 'eq', 'fun':lambda d: np.sum(d[sh.n4l(lmax):])-1}