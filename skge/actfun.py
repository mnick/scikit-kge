import numpy as np
import sys
import inspect


class ActivationFunction(object):

    @classmethod
    def key(cls):
        return cls.__name__.lower()


class Linear(ActivationFunction):

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def g_given_f(fx):
        #return 1
        return np.ones(fx.shape[0])

    # return np.ones((fx.shape[0], 1))


class Sigmoid(ActivationFunction):

    @staticmethod
    def f(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def g_given_f(fx):
        return fx * (1.0 - fx)


class Tanh(ActivationFunction):

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def g_given_f(fx):
        return 1 - fx ** 2


class ReLU(ActivationFunction):

    @staticmethod
    def f(x):
        return np.maximum(0, x)

    @staticmethod
    def g_given_f(fx):
        return np.int_(fx > 0)


class Softplus(ActivationFunction):

    @staticmethod
    def f(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def g(x):
        raise NotImplementedError()


afuns = {}
for cls in ActivationFunction.__subclasses__():
    afuns[cls.key()] = cls
