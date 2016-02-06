# Knowledge Graph Embeddings

scikit-kge is a Python library to compute embeddings of knowledge graphs. The
library consists of different building blocks to train and develop models for
knowledge graph embeddings. These buildings blocks are described in the following:

### Model (skge.base.Model)

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

Training a model with logistic loss function
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
