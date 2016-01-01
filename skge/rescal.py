import numpy as np
from numpy import dot
from skge.base import Model, StochasticTrainer, PairwiseStochasticTrainer
from skge.optim import PredicateAlgorithmMixin, PredicateMarginAlgorithmMixin
from skge.util import grad_sum_matrix, init_nvecs, unzip_triples
from skge.param import ParamInit
import skge.actfun as af
from collections import defaultdict


class RESCAL(Model):

    def __init__(self, *args, **kwargs):
        super(RESCAL, self).__init__(*args, **kwargs)
        self.sz = args[0]
        self.ncomp = args[1]
        self.rparam = kwargs.pop('rparam', 0.0)
        self.init = kwargs.pop('init', 'nunif')
        self.af = kwargs.pop('af', af.Sigmoid)

    def __getstate__(self):
        st = super(RESCAL, self).__getstate__()
        st.update({
            'sz': self.sz,
            'ncomp': self.ncomp,
            'rparam': self.rparam,
            'E': self.E,
            'W': self.W,
            'af': self.af.key(),
        })
        return st

    def _init_factors(self, xs, ys):
        if self.init == 'nvecs':
            self.E, T = init_nvecs(xs, ys, self.sz, self.ncomp, with_T=True)
            self.W = np.array([
                dot(self.E.T, Ti.dot(self.E)) for Ti in T
            ])
        else:
            pinit = ParamInit(self.init)
            self.E = pinit((self.sz[0], self.ncomp))
            self.W = np.array([
                pinit((self.ncomp, self.ncomp)) for _ in range(self.sz[2])
            ])

        self.ups = [
            self.param_updater(self.E, self.learning_rate),
            self.param_updater(self.W, self.learning_rate)
        ]
        #self.E = normalize(self.E)

    def _prepare_model(self):
        self.cache = defaultdict(lambda: {'s': {}, 'o': {}})
        for p, so in self.upmap.items():
            for s in so['s']:
                self.cache[p]['s'][s] = dot(self.E[s], self.W[p])
            for o in so['o']:
                self.cache[p]['o'][o] = dot(self.W[p], self.E[o])

    def _scores(self, ss, ps, os):
        return np.array([
            dot(self.E[ss[i]], dot(self.W[ps[i]], self.E[os[i]]))
            for i in range(len(ss))
        ])

    def _batch_step(self, res):
        ge, gw, eidx, pidx = res

        # object role update
        self.ups[0](ge, eidx)
        self.ups[1](gw, pidx)


class StochasticRESCAL(RESCAL, StochasticTrainer, PredicateAlgorithmMixin):

    def _batch_gradients(self, xys):
        ss, ps, os, ys = unzip_triples(xys, with_ys=True)

        EW = np.array([self.cache[ps[i]]['s'][ss[i]] for i in range(len(ys))])
        WE = np.array([self.cache[ps[i]]['o'][os[i]] for i in range(len(ys))])
        scores = np.sum(self.E[ss] * WE, axis=1)
        preds = af.Sigmoid.f(scores)
        fs = -(ys / (1 + np.exp(ys * scores)))[:, np.newaxis]
        self.loss -= np.sum(ys * np.log(preds))

        #fs = (scores - ys)[:, np.newaxis]
        #self.loss += np.sum(fs * fs)

        pidx = list(self.upmap.keys())
        gw = np.zeros((len(pidx), self.ncomp, self.ncomp))
        for i in range(len(pidx)):
            p = pidx[i]
            ind = np.where(ps == p)[0]
            if len(ind) == 1:
                gw[i] += fs[ind] * np.outer(self.E[ss[ind]], self.E[os[ind]])
            else:
                gw[i] += dot(self.E[ss[ind]].T, fs[ind] * self.E[os[ind]]) / len(ind)
            gw[i] += self.rparam * self.W[p]

        eidx, Sm, n = grad_sum_matrix(list(ss) + list(os))
        ge = Sm.dot(np.vstack((fs * WE, fs * EW))) / n
        ge += self.rparam * self.E[eidx]

        return ge, gw, eidx, pidx


class PairwiseStochasticRESCAL(
        RESCAL,
        PairwiseStochasticTrainer,
        PredicateMarginAlgorithmMixin
):

    def _batch_gradients(self, pxs, nxs):
        # indices of positive examples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative examples
        sn, pn, on = unzip_triples(nxs)

        pxs, _ = np.array(list(zip(*pxs)))
        nxs, _ = np.array(list(zip(*nxs)))

        WEp = np.array([self.cache[p]['o'][o] for _, o, p in pxs])
        WEn = np.array([self.cache[p]['o'][o] for _, o, p in nxs])
        pscores = self.af.f(np.sum(self.E[sp] * WEp, axis=1))
        nscores = self.af.f(np.sum(self.E[sn] * WEn, axis=1))

        #print("avg = %f/%f, min = %f/%f, max = %f/%f" % (pscores.mean(), nscores.mean(), pscores.min(), nscores.min(), pscores.max(), nscores.max()))

        # find examples that violate margin
        ind = np.where(nscores + self.margin > pscores)[0]
        self.nviolations += len(ind)
        if len(ind) == 0:
            return

        # aux vars
        gpscores = -self.af.g_given_f(pscores)[:, np.newaxis]
        gnscores = self.af.g_given_f(nscores)[:, np.newaxis]

        pidx = list(self.upmap.keys())
        gw = np.zeros((len(pidx), self.ncomp, self.ncomp))
        for pid in range(len(pidx)):
            p = pidx[pid]
            ppidx = np.intersect1d(ind, self.pmapp[p])
            npidx = np.intersect1d(ind, self.pmapn[p])
            assert(len(ppidx) == len(npidx))
            if len(ppidx) == 0 and len(npidx) == 0:
                continue
            gw[pid] += dot(self.E[sp[ppidx]].T, gpscores[ppidx] * self.E[op[ppidx]])
            gw[pid] += dot(self.E[sn[npidx]].T, gnscores[npidx] * self.E[on[npidx]])
            gw[pid] += self.rparam * self.W[p]
            gw[pid] /= (len(ppidx) + len(npidx))

        # entity gradients
        sp, sn = list(sp[ind]), list(sn[ind])
        op, on = list(op[ind]), list(on[ind])
        gpscores, gnscores = gpscores[ind], gnscores[ind]
        EWp = np.array([self.cache[p]['s'][s] for s, _, p in pxs[ind]])
        EWn = np.array([self.cache[p]['s'][s] for s, _, p in nxs[ind]])
        eidx, Sm, n = grad_sum_matrix(sp + sn + op + on)
        ge = (Sm.dot(np.vstack((
            gpscores * WEp[ind], gnscores * WEn[ind],
            gpscores * EWp, gnscores * EWn
        ))) + self.rparam * self.E[eidx]) / n

        return ge, gw, eidx, pidx


class RESCALAdaGradL1(RESCAL, PredicateMarginAlgorithmMixin):

    def _init_factors(self, xs, ys):
        super(RESCALAdaGradL1, self)._init_factors(xs, ys)
        self.gE = np.zeros_like(self.E)
        self.gW = np.zeros_like(self.W)
        self.gE2 = np.zeros_like(self.E)
        self.gW2 = np.zeros_like(self.W)
        self.cr = np.zeros(self.W.shape[0])

    def _batch_step(self, res):
        ge, gw, eidx, pidx = res

        # update entities
        #tpn.gE[self.es] += self.gE.data
        self.gE2[eidx] += ge * ge
        #tpn.E[self.es] = ada_threshold_pd(
        #    tpn.gE[self.es],
        #    tpn.gE2[self.es],
        #    tpn.lambdas[0],
        #    tpn.ce[self.es],
        #    tpn.learning_rate)
        #tpn.ce[self.es] += 1

        self.E[eidx] -= ada_step(ge, self.gE2[eidx], self.learning_rate)
        self.E = normalize(self.E, eidx)

        # update predicates
        self.gW[pidx] += gw
        self.gW2[pidx] += gw * gw
        self.W[pidx] = ada_threshold_pd(
            gw,
            self.gW2[pidx],
            self.lambdas[1],
            self.cr[pidx],
            self.learning_rate)
        self.cr[pidx] += 1
