# Knowledge Graph Embeddings

scikit-kge is a Python library to compute embeddings of knowledge graphs. The
library consists of different building blocks to train and develop models for
knowledge graph embeddings.

To compute a knowledge graph embedding, first instantiate a model and then train it
with desired training method. For instance, to train [holographic embeddings of knowledge graphs](http://arxiv.org/abs/1510.04935) (HolE) with a logistcc loss function:

```python
from skge import HolE, StochasticTrainer

# Load knowledge graph 
# N = number of entities
# M = number of relations
# xs = list of (subject, object, predicte) triples
# ys = list of truth values for triples (1 = true, -1 = false)
N, M, xs, ys = load_data('path to data')

# instantiate HolE with an embedding space of size 100
model = HolE((N, N, M), 100)

# instantiate trainer
trainer = StochasticTrainer(model)

# fit model to knowledge graph
trainer.fit(xs, ys)
```

See the [repository for the experiments in the HolE paper](https://github.com/mnick/holographic-embeddings) for an extensive example how to use this library.

The different available buildings blocks are described in more detail in the following:

### Model

Instantiating a model, e.g. HolE
```python
model = HolE(
    self.shape,
    self.args.ncomp,
    init=self.args.init,
    rparam=self.args.rparam
)
```

### Trainer

scikit-kge supports two basic ways to train models: 

##### StochasticTrainer (skge.base.StochasticTrainer)
Trains a model with logistic loss function
```python
trainer = StochasticTrainer(
    model,
    nbatches=100,
    max_epochs=500,
    post_epoch=[self.callback],
    learning_rate=0.1
)
self.trainer.fit(xs, ys)
```
##### PairwiseStochasticTrainer (skge.base)
To train a model with pairwise ranking loss
```python
trainer = PairwiseStochasticTrainer(
    model,
    nbatches=100,
    max_epochs=500,
    post_epoch=[self.callback],
    learning_rate=0.1,
    margin=0.2,
    af=af.Sigmoid
)
self.trainer.fit(xs, ys)
```

### Parameter Update
scitkit-kge supports different methods to update the parameters of a model via
the `param_update` keyword of `StochasticTrainer` and `PairwiseStochasticTrainer`.

For instance,
```python
from skge.param import AdaGrad

trainer = StochasticTrainer(
    ...,
    param_update=AdaGrad,
    ...
)
```
uses `AdaGrad` to update the parameter. 

Available parameter update methods are
##### SGD (skge.param.SGD)
Basic stochastic gradient descent. Only parameter is the learning rate.

##### AdaGrad (skge.param.AdaGrad)
AdaGrad method of [Duchi et al., 2011](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf). Automatically adapts learning rate based on gradient history. Only parameter is the initial learning rate.

### Sampling
sckit-kge implements different strategies to sample negative examples.
