import numpy as np
from skge.base import Model, StochasticTrainer, PairwiseStochasticTrainer
from skge.util import grad_sum_matrix, unzip_triples, ccorr, cconv
from skge.param import ParamInit, normless1
import skge.actfun as af


class HolE(Model):

    def __init__(self, *args, **kwargs):
        super(HolE, self).__init__(*args, **kwargs)
        self.sz = args[0]
        self.ncomp = args[1]
        self.rparam = kwargs.pop('rparam', 0.0)

    def __getstate__(self):
        st = super(HolE, self).__getstate__()
        st.update({
            'sz': self.sz,
            'ncomp': self.ncomp,
            'rparam': self.rparam,
            'E': self.E,
            'R': self.R
        })
        return st

    def _scores(self, ss, ps, os):
        return np.sum(self.R[ps] * ccorr(self.E[ss], self.E[os]), axis=1)

    def _init_factors(self, xs, ys):
        pinit = ParamInit(self.init)
        self.E = pinit((self.sz[0], self.ncomp))
        self.R = pinit((self.sz[2], self.ncomp))
        self.E = normless1(self.E)

        self.ups = [
            #self.param_updater(self.E, self.learning_rate, normless1),
            self.param_updater(self.E, self.learning_rate),
            self.param_updater(self.R, self.learning_rate)
        ]

    def _batch_step(self, res):
        ge, gr, eidx, ridx = res

        # object role update
        self.ups[0](ge, eidx)
        self.ups[1](gr, ridx)


class StochasticHolE(HolE, StochasticTrainer):

    def _batch_gradients(self, xys):
        ss, ps, os, ys = unzip_triples(xys, with_ys=True)

        yscores = ys * self._scores(ss, ps, os)
        self.loss += np.sum(np.logaddexp(0, -yscores))
        #preds = af.Sigmoid.f(yscores)
        fs = -(ys * af.Sigmoid.f(-yscores))[:, np.newaxis]
        #self.loss -= np.sum(np.log(preds))

        ridx, Sm, n = grad_sum_matrix(ps)
        gr = Sm.dot(fs * ccorr(self.E[ss], self.E[os])) / n
        gr += self.rparam * self.R[ridx]

        eidx, Sm, n = grad_sum_matrix(list(ss) + list(os))
        ge = Sm.dot(np.vstack((
            fs * ccorr(self.R[ps], self.E[os]),
            fs * cconv(self.E[ss], self.R[ps])
        ))) / n
        ge += self.rparam * self.E[eidx]

        return ge, gr, eidx, ridx


class PairwiseStochasticHolE(HolE, PairwiseStochasticTrainer):

    def __init__(self, *args, **kwargs):
        super(PairwiseStochasticHolE, self).__init__(*args, **kwargs)
        self.af = kwargs.pop('af', af.Sigmoid)

    def __getstate__(self):
        st = super(PairwiseStochasticHolE, self).__getstate__()
        st.update({'af': self.af.key()})
        return st

    def _batch_gradients(self, pxs, nxs):
        # indices of positive examples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative examples
        sn, pn, on = unzip_triples(nxs)

        pscores = self.af.f(self._scores(sp, pp, op))
        nscores = self.af.f(self._scores(sn, pn, on))

        #print("avg = %f/%f, min = %f/%f, max = %f/%f" % (pscores.mean(), nscores.mean(), pscores.min(), nscores.min(), pscores.max(), nscores.max()))

        # find examples that violate margin
        ind = np.where(nscores + self.margin > pscores)[0]
        self.nviolations += len(ind)
        if len(ind) == 0:
            return

        # aux vars
        sp, sn = list(sp[ind]), list(sn[ind])
        op, on = list(op[ind]), list(on[ind])
        pp, pn = list(pp[ind]), list(pn[ind])
        gpscores = -self.af.g_given_f(pscores[ind])[:, np.newaxis]
        gnscores = self.af.g_given_f(nscores[ind])[:, np.newaxis]

        # object role gradients
        ridx, Sm, n = grad_sum_matrix(pp + pn)
        grp = gpscores * ccorr(self.E[sp], self.E[op])
        grn = gnscores * ccorr(self.E[sn], self.E[on])
        #gr = (Sm.dot(np.vstack((grp, grn))) + self.rparam * self.R[ridx]) / n
        gr = Sm.dot(np.vstack((grp, grn))) / n
        gr += self.rparam * self.R[ridx]

        # filler gradients
        eidx, Sm, n = grad_sum_matrix(sp + sn + op + on)
        geip = gpscores * ccorr(self.R[pp], self.E[op])
        gein = gnscores * ccorr(self.R[pn], self.E[on])
        gejp = gpscores * cconv(self.E[sp], self.R[pp])
        gejn = gnscores * cconv(self.E[sn], self.R[pn])
        ge = Sm.dot(np.vstack((geip, gein, gejp, gejn))) / n
        #ge += self.rparam * self.E[eidx]

        return ge, gr, eidx, ridx
