# Annlite

One can use [AnnLite](https://github.com/jina-ai/annlite) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.


## Usage

### Create DocumentArray with AnnLite backend

One can instantiate a DocumentArray with AnnLite storage like so:

```python
from docarray import DocumentArray

da = DocumentArray(storage='annlite', config={'n_dim': 10})
```

The usage would be the same as the ordinary DocumentArray.

To access a DocumentArray formerly persisted, one can specify the `collection_name`, the `host`  and the `port`.


```python
from docarray import DocumentArray

da = DocumentArray.empty(100, storage='annlite', config={'n_dim': 10, 'metric':'cosine'})

da.summary()
```

Note that specifying the `n_dim` is mandatory before using AnnLite as a backend for DocumentArray.

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name                 | Description                                                                     | Default                              |
|----------------------|---------------------------------------------------------------------------------|--------------------------------------|
| `n_dim`              | Number of dimensions of embeddings to be stored and retrieved                   | **This is always required**          |
| `metric  `           | Distance metric to be used during search. Can be 'cosine',  or 'euclidean'      | 'cosine'                             |
