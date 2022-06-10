(qdrant)=
# Qdrant

One can use [Qdrant](https://qdrant.tech) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `qdrant-client`. You can install it via `pip install "docarray[qdrant]".` 
````

## Usage

### Start Qdrant service

To use Qdrant as the storage backend, you need a running Qdrant server. You can use the Qdrant Docker image to run a 
server. Create `docker-compose.yml` as follows:

```yaml
---
version: '3.4'
services:
  qdrant:
    image: qdrant/qdrant:v0.7.0
    ports:
      - "6333:6333"
    ulimits: # Only required for tests, as there are a lot of collections created
      nofile:
        soft: 65535
        hard: 65535
...
```

Then

```bash
docker-compose up
```

### Create DocumentArray with Qdrant backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:6333`), one can 
instantiate a DocumentArray with Qdrant storage like so:

```python
from docarray import DocumentArray

da = DocumentArray(storage='qdrant', config={'n_dim': 10})
```

The usage would be the same as the ordinary DocumentArray.

To access a DocumentArray formerly persisted, one can specify the `collection_name`, the `host`  and the `port`. 


```python
from docarray import DocumentArray

da = DocumentArray(
    storage='qdrant',
    config={
        'collection_name': 'persisted',
        'host': 'localhost',
        'port': '6333',
        'n_dim': 10,
    },
)

da.summary()
```

Note that specifying the `n_dim` is mandatory before using Qdrant as a backend for DocumentArray.

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name                  | Description                                                                                                                                  | Default                                          |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| `n_dim`               | Number of dimensions of embeddings to be stored and retrieved                                                                                | **This is always required**                      |
| `collection_name`     | Qdrant collection name client                                                                                                                | **Random collection name generated**             |
| `host`                | Hostname of the Qdrant server                                                                                                                | 'localhost'                                      |
| `port`                | port of the Qdrant server                                                                                                                   | 6333                                             |
| `distance`            | Distance metric to be used during search. Can be 'cosine', 'dot' or 'euclidean'                                                              | 'cosine'                                         |
| `scroll_batch_size`   | batch size used when scrolling over the storage                                                                                              | 64                                               |
| `ef_construct`        | Number of neighbours to consider during the index building.  Larger the value - more accurate the search, more time required to build index. | `None`, defaults to the default value in Qdrant* |
| `full_scan_threshold` | Minimal amount of points for additional payload-based indexing.                                                                              | `None`, defaults to the default value in Qdrant*                                               |
| `m`                   | Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.                               | `None`, defaults to the default value in Qdrant*                                               |

*You can read more about the HNSW parameters and their default values [here](https://qdrant.tech/documentation/indexing/#vector-index)

## Minimum example

Create `docker-compose.yml`:

```yaml
---
version: '3.4'
services:
  qdrant:
    image: qdrant/qdrant:v0.7.0
    ports:
      - "6333:6333"
    ulimits: # Only required for tests, as there are a lot of collections created
      nofile:
        soft: 65535
        hard: 65535
...
```

```bash
pip install -U docarray[qdrant]
docker compose up
```


```python
import numpy as np

from docarray import DocumentArray

N, D = 100, 128

da = DocumentArray.empty(N, storage='qdrant', config={'n_dim': D})  # init

da.embeddings = np.random.random([N, D])

print(da.find(np.random.random(D), limit=10))
```


```bash
<DocumentArray (length=10) at 4917906896>
```

(qdrant-filter)=
## Vector search with filter

Search with `.find` can be restricted by user-defined filters. Such filters can be constructed following the guidelines 
in [Qdrant's Documentation](https://qdrant.tech/documentation/filtering/)


### Example of `.find` with a filter


Consider Documents with embeddings `[0,0,0]` up to ` [9,9,9]` where the document with embedding `[i,i,i]`
has as tag `price` with value `i`. We can create such example with the following code:

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
distance = 'euclidean'

da = DocumentArray(
    storage='qdrant',
    config={'n_dim': n_dim, 'columns': [('price', 'float')], 'distance': distance},
)

print(f'\nDocumentArray distance: {distance}')

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )

print('\nIndexed Prices:\n')
for embedding, price in zip(da.embeddings, da[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
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

filter = {'must': [{'key': 'price', 'range': {'lte': max_price}}]}
results = da.find(np_query, filter=filter, limit=n_limit)

print('\nEmbeddings Nearest Neighbours with "price" at most 7:\n')
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

This would print:

```
Query vector: 	[8. 8. 8.]

Embeddings Nearest Neighbours with "price" at most 7:

	embedding=[7. 7. 7.],	 price=7
	embedding=[6. 6. 6.],	 price=6
	embedding=[5. 5. 5.],	 price=5
	embedding=[4. 4. 4.],	 price=4
```
