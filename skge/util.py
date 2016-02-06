import numpy as np
from numpy.fft import fft, ifft
import scipy.sparse as sp
import functools
import collections


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


def to_tensor(xs, ys, sz):
    T = [sp.lil_matrix((sz[0], sz[1])) for _ in range(sz[2])]
    for i in range(len(xs)):
        i, j, k = xs[i]
        T[k][i, j] = ys[i]
    return T


def init_nvecs(xs, ys, sz, rank, with_T=False):
    from scipy.sparse.linalg import eigsh

    T = to_tensor(xs, ys, sz)
    T = [Tk.tocsr() for Tk in T]
    S = sum([T[k] + T[k].T for k in range(len(T))])
    _, E = eigsh(sp.csr_matrix(S), rank)
    if not with_T:
        return E
    else:
        return E, T


class memoized(object):
    '''
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    see https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncachable, return direct function application
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            val = self.func(*args)
            self.cache[args] = val
            return val

    def __repr__(self):
        '''return function's docstring'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''support instance methods'''
        return functools.partial(self.__call__, obj)
