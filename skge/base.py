import numpy as np
from numpy.random import shuffle
from collections import defaultdict
from skge.param import Parameter, AdaGrad
import timeit
import pickle

_cutoff = 30

_DEF_NBATCHES = 100
_DEF_POST_EPOCH = []
_DEF_LEARNING_RATE = 0.1
_DEF_SAMPLE_FUN = None
_DEF_MAX_EPOCHS = 1000
_DEF_MARGIN = 1.0


class Config(object):

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def __getstate__(self):
        return {
            'model': self.model,
            'trainer': self.trainer
        }


class Model(object):
    """
    Base class for all Knowledge Graph models

    Implements basic setup routines for parameters and serialization methods

    Subclasses need to implement:
    - scores(self, ss, ps, os)
    - _gradients(self, xys) for StochasticTrainer
    - _pairwise_gradients(self, pxs, nxs) for PairwiseStochasticTrainer
    """

    def __init__(self, *args, **kwargs):
        #super(Model, self).__init__(*args, **)
        self.params = {}
        self.hyperparams = {}
        self.add_hyperparam('init', kwargs.pop('init', 'nunif'))

    def add_param(self, param_id, shape, post=None, value=None):
        if value is None:
            value = Parameter(shape, self.init, name=param_id, post=post)
        setattr(self, param_id, value)
        self.params[param_id] = value

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def __getstate__(self):
        return {
            'hyperparams': self.hyperparams,
            'params': self.params
        }

    def __setstate__(self, st):
        self.params = {}
        self.hyperparams = {}
        for pid, p in st['params'].items():
            self.add_param(pid, None, None, value=p)
        for pid, p in st['hyperparams'].items():
            self.add_hyperparam(pid, p)

    def save(self, fname, protocol=pickle.HIGHEST_PROTOCOL):
        with open(fname, 'wb') as fout:
            pickle.dump(self, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            mdl = pickle.load(fin)
        return mdl


class StochasticTrainer(object):
    """
    Stochastic gradient descent trainer with scalar loss function.

    Models need to implement

    _gradients(self, xys)

    to be trained with this class.

    """

    def __init__(self, *args, **kwargs):
        self.model = args[0]
        self.hyperparams = {}
        self.add_hyperparam('max_epochs', kwargs.pop('max_epochs', _DEF_MAX_EPOCHS))
        self.add_hyperparam('nbatches', kwargs.pop('nbatches', _DEF_NBATCHES))
        self.add_hyperparam('learning_rate', kwargs.pop('learning_rate', _DEF_LEARNING_RATE))

        self.post_epoch = kwargs.pop('post_epoch', _DEF_POST_EPOCH)
        self.samplef = kwargs.pop('samplef', _DEF_SAMPLE_FUN)
        pu = kwargs.pop('param_update', AdaGrad)
        self._updaters = {
            key: pu(param, self.learning_rate)
            for key, param in self.model.params.items()
        }

    def __getstate__(self):
        return self.hyperparams

    def __setstate__(self, st):
        for pid, p in st['hyperparams']:
            self.add_hyperparam(pid, p)

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def fit(self, xs, ys):
        self._optim(list(zip(xs, ys)))

    def _pre_epoch(self):
        self.loss = 0

    def _optim(self, xys):
        idx = np.arange(len(xys))
        self.batch_size = np.ceil(len(xys) / self.nbatches)
        batch_idx = np.arange(self.batch_size, len(xys), self.batch_size)

        for self.epoch in range(1, self.max_epochs + 1):
            # shuffle training examples
            self._pre_epoch()
            shuffle(idx)

            # store epoch for callback
            self.epoch_start = timeit.default_timer()

            # process mini-batches
            for batch in np.split(idx, batch_idx):
                # select indices for current batch
                bxys = [xys[z] for z in batch]
                self._process_batch(bxys)

            # check callback function, if false return
            for f in self.post_epoch:
                if not f(self):
                    break

    def _process_batch(self, xys):
        # if enabled, sample additional examples
        if self.samplef is not None:
            xys += self.samplef(xys)

        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(xys)

        # take step for batch
        grads = self.model._gradients(xys)
        self.loss += self.model.loss
        self._batch_step(grads)

    def _batch_step(self, grads):
        for paramID in self._updaters.keys():
            self._updaters[paramID](*grads[paramID])


class PairwiseStochasticTrainer(StochasticTrainer):
    """
    Stochastic gradient descent trainer with pairwise ranking loss functions.

    Models need to implement

    _pairwise_gradients(self, pxs, nxs)

    to be trained with this class.

    """


    def __init__(self, *args, **kwargs):
        super(PairwiseStochasticTrainer, self).__init__(*args, **kwargs)
        self.model.add_hyperparam('margin', kwargs.pop('margin', _DEF_MARGIN))

    def fit(self, xs, ys):
        if self.samplef is None:
            pidx = np.where(np.array(ys) == 1)[0]
            nidx = np.where(np.array(ys) != 1)[0]
            pxs = [xs[i] for i in pidx]
            self.nxs = [xs[i] for i in nidx]
            self.pxs = int(len(self.nxs) / len(pxs)) * pxs
            xys = list(range(min(len(pxs), len(self.nxs))))
            self._optim(xys)
        else:
            self._optim(list(zip(xs, ys)))

    def _pre_epoch(self):
        self.nviolations = 0
        if self.samplef is None:
            shuffle(self.pxs)
            shuffle(self.nxs)

    def _process_batch(self, xys):
        pxs = []
        nxs = []

        for xy in xys:
            if self.samplef is not None:
                for nx in self.samplef([xy]):
                    pxs.append(xy)
                    nxs.append(nx)
            else:
                pxs.append((self.pxs[xy], 1))
                nxs.append((self.nxs[xy], 1))

        # take step for batch
        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(pxs, nxs)
        grads = self.model._pairwise_gradients(pxs, nxs)

        # update if examples violate margin
        if grads is not None:
            self.nviolations += self.model.nviolations
            self._batch_step(grads)


def sigmoid(fs):
    # compute elementwise gradient for sigmoid
    for i in range(len(fs)):
        if fs[i] > _cutoff:
            fs[i] = 1.0
        elif fs[i] < -_cutoff:
            fs[i] = 0.0
        else:
            fs[i] = 1.0 / (1 + np.exp(-fs[i]))
    return fs[:, np.newaxis]
