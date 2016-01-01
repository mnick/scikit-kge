import numpy as np
from skge.base import Model, PairwiseStochasticTrainer
from skge.util import grad_sum_matrix, unzip_triples
from skge.param import ParamInit, normalize


class TransE(Model):

    def __init__(self, *args, **kwargs):
        super(TransE, self).__init__(*args, **kwargs)
        self.sz = args[0]
        self.ncomp = args[1]
        self.l1 = kwargs.pop('l1', True)

    def __getstate__(self):
        st = super(TransE, self).__getstate__()
        st.update({
            'sz': self.sz,
            'ncomp': self.ncomp,
            'E': self.E,
            'R': self.R,
            'l1': self.l1
        })
        return st

    def _scores(self, ss, ps, os):
        if self.l1:
            score = np.abs(self.E[ss] + self.R[ps] - self.E[os])
        else:
            score = (self.E[ss] + self.R[ps] - self.E[os]) ** 2
        return -np.sum(score, axis=1)

    def _init_factors(self, xs, ys):
        pinit = ParamInit(self.init)
        self.E = pinit((self.sz[0], self.ncomp))
        self.R = pinit((self.sz[2], self.ncomp))
        self.E = normalize(self.E)
        self.R = normalize(self.R)

        self.ups = [
            self.param_updater(self.E, self.learning_rate, normalize),
            self.param_updater(self.R, self.learning_rate)
        ]

    def _batch_step(self, res):
        ge, gr, eidx, ridx = res
        self.ups[0](ge, eidx)
        self.ups[1](gr, ridx)


class PairwiseStochasticTransE(TransE, PairwiseStochasticTrainer):

    def _batch_gradients(self, pxs, nxs):
        # indices of positive triples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative triples
        sn, pn, on = unzip_triples(nxs)

        pscores = self._scores(sp, pp, op)
        nscores = self._scores(sn, pn, on)
        ind = np.where(nscores + self.margin > pscores)[0]

        # all examples in batch satify margin criterion
        self.nviolations += len(ind)
        if len(ind) == 0:
            return

        sp = list(sp[ind])
        sn = list(sn[ind])
        pp = list(pp[ind])
        pn = list(pn[ind])
        op = list(op[ind])
        on = list(on[ind])

        #pg = self.E[sp] + self.R[pp] - self.E[op]
        #ng = self.E[sn] + self.R[pn] - self.E[on]
        pg = self.E[op] - self.R[pp] - self.E[sp]
        ng = self.E[on] - self.R[pn] - self.E[sn]

        if self.l1:
            pg = np.sign(-pg)
            ng = -np.sign(-ng)
        else:
            raise NotImplementedError()

        # role gradients
        eidx, Sm, n = grad_sum_matrix(sp + op + sn + on)
        ge = Sm.dot(np.vstack((
            pg, -pg, ng, -ng
        ))) / n

        ridx, Sm, n = grad_sum_matrix(pp + pn)
        gr = Sm.dot(np.vstack((pg, ng))) / n
        return ge, gr, eidx, ridx
