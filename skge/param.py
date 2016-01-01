import numpy as np
from numpy import sqrt, squeeze, zeros_like
from numpy.random import randn, uniform


class ParamInit(object):
    """
    Class to initialize weights
    """

    def __init__(self, method):
        if not hasattr(self, method):
            raise ValueError('Unknown initialization (%s)' % method)
        self.method = method

    def __call__(self, sz):
        if len(sz) != 2:
            raise ValueError('Shape must be of size 2')
        return getattr(self, self.method)(sz)

    def unif(self, sz):
        """
        Uniform intialization

        Heuristic commonly used to initialize deep neural networks
        """
        bnd = 1 / sqrt(sz[0])
        p = uniform(low=-bnd, high=bnd, size=sz)
        return squeeze(p)

    def nunif(self, sz):
        """
        Normalized uniform initialization

        See Glorot X., Bengio Y.: "Understanding the difficulty of training
        deep feedforward neural networks". AISTATS, 2010
        """
        bnd = sqrt(6) / sqrt(sz[0] + sz[1])
        p = uniform(low=-bnd, high=bnd, size=sz)
        return squeeze(p)

    def randn(self, sz):
        return squeeze(randn(*sz))


class SGDUpdate(object):
    """
    Class to perform SGD updates on a parameter
    """

    def __init(self, param, learning_rate):
        self.param = param
        self.learning_rate = learning_rate

    def __call__(self, g, idx=None):
        self.param[idx] -= self.learning_rate * g

    def reset(self):
        pass


class AdaGradUpdate(object):

    def __init__(self, param, learning_rate, post_update=None):
        self.param = param
        self.learning_rate = learning_rate
        self.p2 = zeros_like(param)
        self.post_udpate = post_update

    def __call__(self, g, idx=None):
        self.p2[idx] += g * g
        H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
        self.param[idx] -= self.learning_rate * g / H
        if self.post_udpate is not None:
            self.param = self.post_udpate(self.param, idx)

    def reset(self):
        self.p2 = zeros_like(self.p2)


def normalize(M, idx=None):
    if idx is None:
        M = M / np.sqrt(np.sum(M ** 2, axis=1))[:, np.newaxis]
    else:
        nrm = np.sqrt(np.sum(M[idx, :] ** 2, axis=1))[:, np.newaxis]
        M[idx, :] = M[idx, :] / nrm
    return M


def normless1(M, idx=None):
    nrm = np.sum(M[idx] ** 2, axis=1)[:, np.newaxis]
    nrm[nrm < 1] = 1
    M[idx] = M[idx] / nrm
    return M
