# Knowledge Graph Embeddings

scikit-kge is a Python library to compute embeddings of knowledge graphs

Instantiating a model, e.g. HolE
```python
model = HolE(
    self.shape,
    self.args.ncomp,
    init=self.args.init,
    rparam=self.args.rparam
)
```

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

### Sampling
sckit-kge implements different strategies to sample negative examples
