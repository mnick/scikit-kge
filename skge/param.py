import sys
import numpy as np
from numpy import sqrt, squeeze, zeros_like
from numpy.random import randn, uniform


def init_unif(sz):
        """
        Uniform intialization

        Heuristic commonly used to initialize deep neural networks
        """
        bnd = 1 / sqrt(sz[0])
        p = uniform(low=-bnd, high=bnd, size=sz)
        return squeeze(p)


def init_nunif(sz):
        """
        Normalized uniform initialization

        See Glorot X., Bengio Y.: "Understanding the difficulty of training
        deep feedforward neural networks". AISTATS, 2010
        """
        bnd = sqrt(6) / sqrt(sz[0] + sz[1])
        p = uniform(low=-bnd, high=bnd, size=sz)
        return squeeze(p)


def init_randn(sz):
        return squeeze(randn(*sz))


class Parameter(np.ndarray):

    def __new__(cls, *args, **kwargs):
        # TODO: hackish, find better way to handle higher-order parameters
        if len(args[0]) == 3:
                sz = (args[0][1], args[0][2])
                arr = np.array([Parameter._init_array(sz, args[1]) for _ in range(args[0][0])])
        else:
                arr = Parameter._init_array(args[0], args[1])
        arr = arr.view(cls)
        arr.name = kwargs.pop('name', None)
        arr.post = kwargs.pop('post', None)

        if arr.post is not None:
            arr = arr.post(arr)

        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.post = getattr(obj, 'post', None)

    @staticmethod
    def _init_array(shape, method):
        mod = sys.modules[__name__]
        method = 'init_%s' % method
        if not hasattr(mod, method):
            raise ValueError('Unknown initialization (%s)' % method)
        elif len(shape) != 2:
            raise ValueError('Shape must be of size 2')
        return getattr(mod, method)(shape)


class ParameterUpdate(object):

    def __init__(self, param, learning_rate):
        self.param = param
        self.learning_rate = learning_rate

    def __call__(self, gradient, idx=None):
        self._update(gradient, idx)
        if self.param.post is not None:
            self.param = self.param.post(self.param, idx)

    def reset(self):
        pass


class SGD(ParameterUpdate):
    """
    Class to perform SGD updates on a parameter
    """

    def _update(self, g, idx):
        self.param[idx] -= self.learning_rate * g


class AdaGrad(ParameterUpdate):

    def __init__(self, param, learning_rate):
        super(AdaGrad, self).__init__(param, learning_rate)
        self.p2 = zeros_like(param)

    def _update(self, g, idx=None):
        self.p2[idx] += g * g
        H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
        self.param[idx] -= self.learning_rate * g / H

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
