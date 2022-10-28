# Annlite

One can use [Annlite](https://github.com/jina-ai/annlite) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `annlite`. You can install it via `pip install "docarray[full]".` 
````


## Usage

One can instantiate a DocumentArray with Annlite storage like so:

```python
from docarray import DocumentArray

da = DocumentArray(storage='annlite', config={'n_dim': 10})
```

The usage would be the same as the ordinary DocumentArray.

To access a DocumentArray formerly persisted, one can specify the `data_path` in `config`. 

```python
from docarray import DocumentArray

da = DocumentArray(storage='annlite', config={'data_path': './data', 'n_dim': 10})

da.summary()
```

Note that specifying the `n_dim` is mandatory before using `Annlite` as a backend for DocumentArray.

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name                | Description                                                                     | Default                     |
|---------------------|---------------------------------------------------------------------------------|-----------------------------|
| `n_dim`             | Number of dimensions of embeddings to be stored and retrieved                   | **This is always required** |
| `data_path`         | The data folder where the data is located                                       | **A random temp folder**    |
| `metric`            | Distance metric to be used during search. Can be 'cosine', 'dot' or 'euclidean' | 'cosine'                    |
