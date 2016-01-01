import numpy as np
from numpy.fft import fft, ifft
import scipy.sparse as sp


def cconv(a, b):
    """
    Circular convolution of vectors

    Computes the circular convolution of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\mathcal{F}(a) \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       convolution of a and b
    """
    return ifft(fft(a) * fft(b)).real


def ccorr(a, b):
    """
    Circular correlation of vectors

    Computes the circular correlation of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\overline{\mathcal{F}(a)} \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       correlation of a and b
    """

    return ifft(np.conj(fft(a)) * fft(b)).real


def grad_sum_matrix(idx):
    uidx, iinv = np.unique(idx, return_inverse=True)
    sz = len(iinv)
    M = sp.coo_matrix((np.ones(sz), (iinv, np.arange(sz)))).tocsr()
    # normalize summation matrix so that each row sums to one
    n = np.array(M.sum(axis=1))
    #M = M.T.dot(np.diag(n))
    return uidx, M, n


def unzip_triples(xys, with_ys=False):
    xs, ys = list(zip(*xys))
    ss, os, ps = list(zip(*xs))
    if with_ys:
        return np.array(ss), np.array(ps), np.array(os), np.array(ys)
    else:
        return np.array(ss), np.array(ps), np.array(os)
