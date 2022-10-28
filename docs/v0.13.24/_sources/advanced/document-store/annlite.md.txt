(annlite)=
# Annlite

One can use [Annlite](https://github.com/jina-ai/annlite) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `annlite`. You can install it via `pip install "docarray[annlite]".` 
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

| Name              | Description                                                                           | Default                                                       |
|-------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `n_dim`           | Number of dimensions of embeddings to be stored and retrieved                         | **This is always required**                                   |
| `data_path`       | The data folder where the data is located                                             | **A random temp folder**                                      |
| `metric`          | Distance metric to be used during search. Can be 'cosine', 'dot' or 'euclidean'       | 'cosine'                                                      |
| `ef_construction` | The size of the dynamic list for the nearest neighbors (used during the construction) | `None`, defaults to the default value in the AnnLite package* |
| `ef_search`       | The size of the dynamic list for the nearest neighbors (used during the search)       | `None`, defaults to the default value in the AnnLite package* |
| `max_connection`  | The number of bi-directional links created for every new element during construction. | `None`, defaults to the default value in the AnnLite package* |

*You can check the default values in [the AnnLite source code](https://github.com/jina-ai/annlite/blob/main/annlite/core/index/hnsw/index.py)

(annlite-filter)=
## Vector search with filter

Search with `.find` can be restricted by user-defined filters.
Filters can be constructed following the guidelines provided in [the AnnLite source repository](https://github.com/jina-ai/annlite).

### Example of `.find` with a filter only


Consider you store Documents with a certain tag `price` into annlite and you want to retrieve all Documents
with `price`  lower or equal to  some `max_price` value.


You can index such Documents as follows:
```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
da = DocumentArray(
    storage='annlite',
    config={
        'n_dim': n_dim,
        'columns': [('price', 'float')],
    },
)

with da:
    da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])

print('\nIndexed Prices:\n')
for price in da[:, 'tags__price']:
    print(f'\t price={price}')
```

Then you can retrieve all documents whose price is lower than or equal to `max_price` by applying the following
filter:

```python
max_price = 3
n_limit = 4

filter = {'price': {'$lte': max_price}}
results = da.find(filter=filter)

print('\n Returned examples that verify filter "price at most 3":\n')
for price in results[:, 'tags__price']:
    print(f'\t price={price}')
```

This would print

```
 Returned examples that satisfy condition "price at most 3":

	 price=0
	 price=1
	 price=2
	 price=3
```

### Example of `.find` with query vector and filter

Consider Documents with embeddings `[0,0,0]` up to ` [9,9,9]` where the document with embedding `[i,i,i]`
has as tag `price` with value `i`. We can create such example with the following code:


```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
metric = 'Euclidean'

da = DocumentArray(
    storage='annlite',
    config={'n_dim': n_dim, 'columns': [('price', 'float')], 'metric': metric},
)

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )
```

Consider we want the nearest vectors to the embedding `[8. 8. 8.]`, with the restriction that
prices must follow a filter. As an example, let's consider that retrieved documents must have `price` value lower
or equal than `max_price`. We can encode this information in annlite using `filter = {'price': {'$lte': max_price}}`.

Then the search with the proposed filter can be implemented and used with the following code:

```python
max_price = 7
n_limit = 4

np_query = np.ones(n_dim) * 8
print(f'\nQuery vector: \t{np_query}')

filter = {'price': {'$lte': max_price}}
results = da.find(np_query, filter=filter, limit=n_limit)
print('\nEmbeddings Nearest Neighbours with "price" at most 7:\n')
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

This would print:

```bash
Query vector: 	[8. 8. 8.]

Embeddings Nearest Neighbours with "price" at most 7:

	embedding=[7. 7. 7.],	 price=7
	embedding=[6. 6. 6.],	 price=6
	embedding=[5. 5. 5.],	 price=5
	embedding=[4. 4. 4.],	 price=4
 ```
