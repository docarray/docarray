(qdrant)=
# Qdrant

You can use [Qdrant](https://qdrant.tech) as a document store for DocumentArray. It's suitable for faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `qdrant-client`. You can install it with `pip install "docarray[qdrant]".` 
````

## Usage

### Start Qdrant service

To use Qdrant as the storage backend, you need a running Qdrant server. You can create `docker-compose.yml` to use the Qdrant Docker image:

```yaml
---
version: '3.4'
services:
  qdrant:
    image: qdrant/qdrant:v0.10.1
    ports:
      - "6333:6333"
      - "6334:6334"
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

Assuming you start the service with the default configuration (i.e. server address is `http://localhost:6333`), you can 
instantiate a DocumentArray with Qdrant storage like so:

```python
from docarray import DocumentArray

da = DocumentArray(storage='qdrant', config={'n_dim': 10})
```

The usage is the same as an ordinary DocumentArray.

To access a formerly-persisted DocumentArray, you can specify the `collection_name`, `host` and `port`:


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

Note that you must specify `n_dim` before using Qdrant as a backend for DocumentArray.

Other functions behave the same as an in-memory DocumentArray.

## Configuration

| Name                  | Description                                                                                                                            | Default                                          |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| `n_dim`               | Number of dimensions of embeddings to be stored and retrieved                                                                          | **This is always required**                      |
| `collection_name`     | Qdrant collection name client                                                                                                          | **Random collection name generated**             |
| `distance`            | Distance metric to use during search. Can be 'cosine' (similarity), 'dot' or 'euclidean'                                               | `'cosine'`                                       |
| `host`                | Hostname of the Qdrant server                                                                                                          | `'localhost'`                                    |
| `port`                | Port of the Qdrant server                                                                                                              | `6333`                                           |
| `grpc_port`           | Port of the Qdrant gRPC interface                                                                                                      | `6334`                                           |
| `prefer_grpc`         | Set `True` to use gPRC interface whenever possible in custom methods                                                                   | `False`                                          |
| `api_key`             | API key for authentication in Qdrant Cloud                                                                                             | `None`                                           |
| `https`               | Set `True` to use HTTPS(SSL) protocol                                                                                                  | `None`                                           |
| `serialize_config`    | [Serialization config of each Document](../../../fundamentals/document/serialization.md)                                               | `None`                                           |
| `scroll_batch_size`   | Batch size used when scrolling over the storage                                                                                        | `64`                                             |
| `ef_construct`        | Number of neighbours to consider during the index building.  Larger = more accurate search, more time to build index                   | `None`, defaults to the default value in Qdrant* |
| `full_scan_threshold` | Minimal size (in KiloBytes) of vectors for additional payload-based indexing                                                           | `None`, defaults to the default value in Qdrant* |
| `m`                   | Number of edges per node in the index graph. Larger = more accurate search, more space required                                        | `None`, defaults to the default value in Qdrant* |
| `columns`             | Other fields to store in Document                                                                                                      | `None`                                           |
| `list_like`           | Controls if ordering of Documents is persisted in the Database. Disabling this breaks list-like features, but can improve performance. | True                                             |
| `root_id`             | Boolean flag indicating whether to store `root_id` in the tags of chunk level Documents                                                | True                                             |


*You can read more about the HNSW parameters and their default values [here](https://qdrant.tech/documentation/indexing/#vector-index)

## Minimum example

Create `docker-compose.yml`:

```yaml
---
version: '3.4'
services:
  qdrant:
    image: qdrant/qdrant:v0.10.1
    ports:
      - "6333:6333"
      - "6334:6334"
    ulimits: # Only required for tests, as there are a lot of collections created
      nofile:
        soft: 65535
        hard: 65535
...
```

```bash
pip install -U docarray[qdrant]
docker-compose up
```


```python
import numpy as np

from docarray import DocumentArray

N, D = 100, 128

da = DocumentArray.empty(
    N, storage='qdrant', config={'n_dim': D, 'distance': 'cosine'}
)  # init

da.embeddings = np.random.random([N, D])

print(da.find(np.random.random(D), limit=10))
```


```bash
<DocumentArray (length=10) at 4917906896>
```

(qdrant-filter)=
## Vector search with filter

Search with `.find` can be restricted by user-defined filters. The supported tag types for filter are `'int'`, `'float'`, `'bool'`, `'str'`, `'text'` and `'geo'` as in [Qdrant](https://qdrant.tech/documentation/payload/). Such filters can be constructed following the guidelines in [Qdrant's Documentation](https://qdrant.tech/documentation/filtering/)



### Example of `.find` with filter

Let's create Documents with embeddings `[0,0,0]` up to `[9,9,9]`, where each Document (which has an embedding `[i,i,i]`)
has a tag `price` with value `i`:

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
distance = 'euclidean'

da = DocumentArray(
    storage='qdrant',
    config={'n_dim': n_dim, 'columns': {'price': 'float'}, 'distance': distance},
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

We want the nearest vectors to the embedding `[8. 8. 8.]`, with the restriction that prices must follow a filter. For example, retrieved Documents must have `price` value lower than or equal to `max_price`. You can encode this information in Qdrant using `filter = {'must': [{'key': 'price', 'range': {'lte': max_price}}]}`. You can also pass additional `search_params` following [Qdrant's Search API](https://qdrant.tech/documentation/search/#search-api).

You can then implement and search with the proposed filter:

```python
max_price = 7
n_limit = 4

np_query = np.ones(n_dim) * 8
print(f'\nQuery vector: \t{np_query}')

filter = {'must': [{'key': 'price', 'range': {'lte': max_price}}]}
results = da.find(np_query, filter=filter, limit=n_limit, search_params={"hnsw_ef": 64})

print('\nEmbeddings Nearest Neighbours with "price" at most 7:\n')
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

This prints:

```
Query vector: 	[8. 8. 8.]

Embeddings Nearest Neighbours with "price" at most 7:

	embedding=[7. 7. 7.],	 price=7
	embedding=[6. 6. 6.],	 price=6
	embedding=[5. 5. 5.],	 price=5
	embedding=[4. 4. 4.],	 price=4
```

````{admonition} Note
:class: note
For Qdrant, the distance scores can be accessed in the Document's `.scores` dictionary by the key `f'{distance_metric}_similarity'`. For example, for `distance = 'euclidean'` the key would be `'euclidean_similarity'`.
````

### Example of `.filter` with a filter

The following example shows how to use DocArray with Qdrant document store to filter text documents.
Let's create Documents with the tag `price` with a value of `i`:
```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3

da = DocumentArray(
    storage='qdrant',
    config={'n_dim': n_dim, 'columns': {'price': 'float'}},
)

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

If you want to filter only for results 
with a `price` less than or equal to `max_price`, you can encode 
this information using `filter = {'price': {'$lte': max_price}}`.

You can then implement and search with the proposed filter:

```python
max_price = 7
n_limit = 4

filter = {'must': [{'key': 'price', 'range': {'lte': max_price}}]}
results = da.filter(filter=filter, limit=n_limit)

print('\nPoints with "price" at most 7:\n')
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```
This prints:

```

Points with "price" at most 7:

	embedding=[6. 6. 6.],	 price=6
	embedding=[7. 7. 7.],	 price=7
	embedding=[1. 1. 1.],	 price=1
	embedding=[2. 2. 2.],	 price=2
```
