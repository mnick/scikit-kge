import numpy as np
from numpy.random import shuffle
import scipy.sparse as sp
from collections import defaultdict
from skge.param import AdaGradUpdate
import timeit
import pickle

_cutoff = 30

_DEF_NBATCHES = 100
_DEF_CALLBACK = None
_DEF_LEARNING_RATE = 0.1
_DEF_SAMPLE_FUN = None
_DEF_MAX_EPOCHS = 1000
_DEF_MARGIN = 1.0


class Model(object):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

    def __getstate__(self):
        return {
            'max_epochs': self.max_epochs,
            'nbatches': self.nbatches,
            'learning_rate': self.learning_rate,
            'init': self.init,
            'epoch': self.epoch
        }

    def __setstate__(self, st):
        self.__dict__ = st

    def save(self, fname, protocol=pickle.HIGHEST_PROTOCOL):
        with open(fname, 'wb') as fout:
            pickle.dump(self, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            mdl = pickle.load(fin)
        return mdl


class StochasticTrainer(object):

    def __init__(self, *args, **kwargs):
        self.max_epochs = kwargs.pop('max_epochs', _DEF_MAX_EPOCHS)
        self.nbatches = kwargs.pop('nbatches', _DEF_NBATCHES)
        self.callback = kwargs.pop('callback', _DEF_CALLBACK)
        self.learning_rate = kwargs.pop('learning_rate', _DEF_LEARNING_RATE)
        self.samplef = kwargs.pop('samplef', _DEF_SAMPLE_FUN)
        self.param_updater = kwargs.pop('param_update', AdaGradUpdate)
        self.init = kwargs.pop('init', 'randn')

    def fit(self, xs, ys):
        self._init_factors(xs, ys)
        self._optim(list(zip(xs, ys)))

    def pre_epoch(self):
        self.loss = 0

    def _optim(self, xys):
        idx = np.arange(len(xys))
        self.batch_size = np.ceil(len(xys) / self.nbatches)
        batch_idx = np.arange(self.batch_size, len(xys), self.batch_size)

        for self.epoch in range(1, self.max_epochs + 1):
            # shuffle training examples
            self.pre_epoch()
            shuffle(idx)

            # store epoch for callback
            self.epoch_start = timeit.default_timer()

            # process mini-batches
            for batch in np.split(idx, batch_idx):
                # select indices for current batch
                bxys = [xys[z] for z in batch]
                self._process_batch(bxys)

            # check callback function, if false return
            if self.callback is not None and not self.callback(self):
                break

    def _process_batch(self, xys):
        # if enabled, sample additional examples
        if self.samplef is not None:
            xys += self.samplef(xys)

        if hasattr(self, '_prepare_batch_step'):
            self._prepare_batch_step(xys)

        # take step for batch
        res = self._batch_gradients(xys)
        self._batch_step(res)


class PairwiseStochasticTrainer(StochasticTrainer):

    def __init__(self, *args, **kwargs):
        super(PairwiseStochasticTrainer, self).__init__(*args, **kwargs)
        self.margin = kwargs.pop('margin', _DEF_MARGIN)

    def __getstate__(self):
        st = super(PairwiseStochasticTrainer, self).__getstate__()
        st.update({'margin': self.margin})
        return st

    def pre_epoch(self):
        self.nviolations = 0

    def _process_batch(self, xys):
        pxs = []
        nxs = []

        for xy in xys:
            if self.samplef is not None:
                for nx in self.samplef([xy]):
                    pxs.append(xy)
                    nxs.append(nx)
            else:
                if xy[1] == 1:
                    pxs.append((xy[0], 1))
                else:
                    nxs.append((xy[0], -1))

        if self.samplef is None:
            pidx, nidx = np.arange(len(pxs)), np.arange(len(nxs))
            pidx, nidx = np.meshgrid(pidx, nidx)
            pxs = [pxs[i] for i in pidx.flatten()]
            nxs = [nxs[i] for i in nidx.flatten()]

        # take step for batch
        if hasattr(self, '_prepare_batch_step'):
            self._prepare_batch_step(pxs, nxs)
        res = self._batch_gradients(pxs, nxs)

        # update if examples violate margin
        if res is not None:
            self._batch_step(res)


class PredicateAlgorithmMixin:

    def _prepare_triples(self, xys):
        pmap = defaultdict(list)
        upmap = defaultdict(lambda: {'s': set(), 'o': set()})
        for i in range(len(xys)):
            x, y = xys[i]
            s, o, p = x
            pmap[p].append(i)
            upmap[p]['s'].add(s)
            upmap[p]['o'].add(o)
        return dict(pmap), dict(upmap)

    def _prepare_batch_step(self, xs):
        self.pmap, self.upmap = self._prepare_triples(xs)
        self._prepare_model()


class PredicateMarginAlgorithmMixin(PredicateAlgorithmMixin):

    def _prepare_batch_step(self, pxs, nxs):
        self.pmapp, self.upmap = self._prepare_triples(pxs)
        self.pmapn, upmapn = self._prepare_triples(nxs)
        for p, so in upmapn.items():
            self.upmap[p]['s'] = self.upmap[p]['s'].union(so['s'])
            self.upmap[p]['o'] = self.upmap[p]['o'].union(so['o'])
        self._prepare_model()


# def ada_step(g, g2, eta):
#     H = np.maximum(np.sqrt(g2), 1e-6)
#     return eta * g / H


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
