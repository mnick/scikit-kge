import numpy as np
from numpy import dot
from skge.base import Model
from skge.base import PredicateAlgorithmMixin
from skge.util import grad_sum_matrix, unzip_triples
import skge.actfun as af
from collections import defaultdict


class RESCAL(Model):
    """
    Base class for RESCAL

    Use either
    - StochasticRESCAL: to train via RESCAL via logistic loss
    - PairwiseStochasticRESCAL: to train RESCAL via ranking loss
    """

    def __init__(self, *args, **kwargs):
        super(RESCAL, self).__init__(*args, **kwargs)
        self.add_hyperparam('sz', args[0])
        self.add_hyperparam('ncomp', args[1])
        self.add_hyperparam('rparam', kwargs.pop('rparam', 0.0))
        self.add_hyperparam('af', kwargs.pop('af', af.Sigmoid))
        self.add_param('E', (self.sz[0], self.ncomp))
        # TODO: support eigenvector initialization
        self.add_param('W', (self.sz[2], self.ncomp, self.ncomp))

    # TODO: this is not used right now
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

    def _gradients(self, xys):
        ss, ps, os, ys = unzip_triples(xys, with_ys=True)

        #EW = np.array([self.cache[ps[i]]['s'][ss[i]] for i in range(len(ys))])
        #WE = np.array([self.cache[ps[i]]['o'][os[i]] for i in range(len(ys))])
        EW = np.array([dot(self.E[s], self.W[p]) for ((s, _, p), _) in xys])
        WE = np.array([dot(self.W[p], self.E[o]) for ((_, o, p), _) in xys])
        yscores = ys * np.sum(self.E[ss] * WE, axis=1)
        self.loss = np.sum(np.logaddexp(0, -yscores))
        fs = -(ys * af.Sigmoid.f(-yscores))[:, np.newaxis]
        #preds = af.Sigmoid.f(scores)
        #fs = -(ys / (1 + np.exp(yscores)))[:, np.newaxis]
        #self.loss -= np.sum(ys * np.log(preds))

        #fs = (scores - ys)[:, np.newaxis]
        #self.loss += np.sum(fs * fs)

        #pidx = list(self.upmap.keys())
        pidx = np.unique(ps)
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

        return {'E': (ge, eidx), 'W': (gw, pidx)}

    def _pairwise_gradients(self, pxs, nxs):
        # indices of positive examples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative examples
        sn, pn, on = unzip_triples(nxs)

        pxs, _ = np.array(list(zip(*pxs)))
        nxs, _ = np.array(list(zip(*nxs)))


        #WEp = np.array([self.cache[p]['o'][o] for _, o, p in pxs])
        #WEn = np.array([self.cache[p]['o'][o] for _, o, p in nxs])
        WEp = np.array([dot(self.W[p], self.E[o]) for _, o, p in pxs])
        WEn = np.array([dot(self.W[p], self.E[o]) for _, o, p in nxs])
        pscores = self.af.f(np.sum(self.E[sp] * WEp, axis=1))
        nscores = self.af.f(np.sum(self.E[sn] * WEn, axis=1))

        #print("avg = %f/%f, min = %f/%f, max = %f/%f" % (pscores.mean(), nscores.mean(), pscores.min(), nscores.min(), pscores.max(), nscores.max()))

        # find examples that violate margin
        ind = np.where(nscores + self.margin > pscores)[0]
        self.nviolations = len(ind)
        if len(ind) == 0:
            return

        # aux vars
        gpscores = -self.af.g_given_f(pscores)[:, np.newaxis]
        gnscores = self.af.g_given_f(nscores)[:, np.newaxis]

        pidx = np.unique(list(pp) + list(pn))
        gw = np.zeros((len(pidx), self.ncomp, self.ncomp))
        for pid in range(len(pidx)):
            p = pidx[pid]
            #ppidx = np.intersect1d(ind, self.pmapp[p])
            #npidx = np.intersect1d(ind, self.pmapn[p])
            ppidx = np.where(pp == p)
            npidx = np.where(pn == p)
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
        # EWp = np.array([self.cache[p]['s'][s] for s, _, p in pxs[ind]])
        # EWn = np.array([self.cache[p]['s'][s] for s, _, p in nxs[ind]])
        EWp = np.array([dot(self.E[s], self.W[p]) for s, _, p in pxs])
        EWn = np.array([dot(self.E[s], self.W[p]) for s, _, p in nxs])
        eidx, Sm, n = grad_sum_matrix(sp + sn + op + on)
        ge = (Sm.dot(np.vstack((
            gpscores * WEp[ind], gnscores * WEn[ind],
            gpscores * EWp, gnscores * EWn
        ))) + self.rparam * self.E[eidx]) / n

        return {'E': (ge, eidx), 'W': (gw, pidx)}
