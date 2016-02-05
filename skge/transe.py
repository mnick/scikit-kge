import numpy as np
from skge.base import Model
from skge.util import grad_sum_matrix, unzip_triples
from skge.param import normalize


class TransE(Model):
    """
    Translational Embeddings of Knowledge Graphs
    """

    def __init__(self, *args, **kwargs):
        super(TransE, self).__init__(*args, **kwargs)
        self.add_hyperparam('sz', args[0])
        self.add_hyperparam('ncomp', args[1])
        self.add_hyperparam('l1', kwargs.pop('l1', True))
        self.add_param('E', (self.sz[0], self.ncomp), post=normalize)
        self.add_param('R', (self.sz[2], self.ncomp))

    def _scores(self, ss, ps, os):
        score = self.E[ss] + self.R[ps] - self.E[os]
        if self.l1:
            score = np.abs(score)
        else:
            score = score ** 2
        return -np.sum(score, axis=1)

    def _pairwise_gradients(self, pxs, nxs):
        # indices of positive triples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative triples
        sn, pn, on = unzip_triples(nxs)

        pscores = self._scores(sp, pp, op)
        nscores = self._scores(sn, pn, on)
        ind = np.where(nscores + self.margin > pscores)[0]

        # all examples in batch satify margin criterion
        self.nviolations = len(ind)
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

        # entity gradients
        eidx, Sm, n = grad_sum_matrix(sp + op + sn + on)
        ge = Sm.dot(np.vstack((pg, -pg, ng, -ng))) / n

        # relation gradients
        ridx, Sm, n = grad_sum_matrix(pp + pn)
        gr = Sm.dot(np.vstack((pg, ng))) / n
        return {'E': (ge, eidx), 'R': (gr, ridx)}
