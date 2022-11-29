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

| Name                  | Description                                                                                                                              | Default                                          |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| `n_dim`               | Number of dimensions of embeddings to be stored and retrieved                                                                            | **This is always required**                      |
| `collection_name`     | Qdrant collection name client                                                                                                            | **Random collection name generated**             |
| `distance`            | Distance metric to use during search. Can be 'cosine', 'dot' or 'euclidean'                                                              | `'cosine'`                                       |
| `host`                | Hostname of the Qdrant server                                                                                                            | `'localhost'`                                    |
| `port`                | Port of the Qdrant server                                                                                                                | `6333`                                           |
| `grpc_port`           | Port of the Qdrant gRPC interface                                                                                                        | `6334`                                           |
| `prefer_grpc`         | Set `True` to use gPRC interface whenever possible in custom methods                                                                     | `False`                                          |
| `api_key`             | API key for authentication in Qdrant Cloud                                                                                               | `None`                                           |
| `https`               | Set `True` to use HTTPS(SSL) protocol                                                                                                    | `None`                                           |
| `serialize_config`    | [Serialization config of each Document](../../../fundamentals/document/serialization.md)                                                 | `None`                                           |
| `index_text`          | Boolean flag indicating whether to index `.text`. `True` will enable full text search on `.text`                                         | `None`                                           |
| `tag_indices`         | List of tags to index as text field                                                                                                      | `[]`                                             |
| `scroll_batch_size`   | Batch size used when scrolling over the storage                                                                                          | `64`                                             |
| `ef_construct`        | Number of neighbours to consider during the index building.  Larger = more accurate search, more time to build index                     | `None`, defaults to the default value in Qdrant* |
| `full_scan_threshold` | Minimal size (in KiloBytes) of vectors for additional payload-based indexing                                                             | `None`, defaults to the default value in Qdrant* |
| `m`                   | Number of edges per node in the index graph. Larger = more accurate search, more space required                                          | `None`, defaults to the default value in Qdrant* |
| `columns`             | Other fields to store in Document                                                                                                        | `None`                                           |
| `list_like`           | Controls if ordering of Documents is persisted in the Database. Disabling this breaks list-like features, but can improve performance.   | `True`                                           |
| `root_id`             | Boolean flag indicating whether to store `root_id` in the tags of chunk level Documents                                                  | `True`                                           |


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

Consider we want the nearest vectors to the embedding `[8. 8. 8.]`, with the restriction that prices must follow a filter. As an example, retrieved Documents must have `price` value lower than or equal to `max_price`. We can encode this information in Qdrant using `filter = {'must': [{'key': 'price', 'range': {'lte': max_price}}]}`. You can also pass additional `search_params` following [Qdrant's Search API](https://qdrant.tech/documentation/search/#search-api).

Then you can implement and use the search with the proposed filter:

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

This would print:

```
Query vector: 	[8. 8. 8.]

Embeddings Nearest Neighbours with "price" at most 7:

	embedding=[7. 7. 7.],	 price=7
	embedding=[6. 6. 6.],	 price=6
	embedding=[5. 5. 5.],	 price=5
	embedding=[4. 4. 4.],	 price=4
```
### Example of `.filter` with a filter
The following example shows how to use DocArray with Qdrant Document Store in order to filter text documents.
Consider Documents have the tag `price` with a value of `i`. We can create these with the following code:
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
For example, suppose we want to filter results such that
retrieved documents must have a `price` value less than or equal to `max_price`. We can encode 
this information in Qdrant using `filter = {'price': {'$lte': max_price}}`.

Then you can implement and use the search with the proposed filter:
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


### Search by `.text` field

You can perform full-text search in a `DocumentArray` with `storage='qdrant'`. 
To do this, text needs to be indexed using the boolean flag `'index_text'` which is set when the `DocumentArray` is created  with `config={'index_text': True, ...}`.
The following example builds a `DocumentArray` with several documents containing text and searches for those that have `token1` in their text description.

```python
from docarray import Document, DocumentArray

da = DocumentArray(storage='qdrant', config={'n_dim': 2, 'index_text': True})
with da:
    da.extend(
        [
            Document(id='1', text='token1 token2 token3'),
            Document(id='2', text='token1 token2'),
            Document(id='3', text='token2 token3 token4'),
        ]
    )

results = da.find('token1')
print(results[:, 'text'])
```

This will print:

```console
['token1 token2 token3', 'token1 token2']
```

According to [Qdrant](https://qdrant.tech/documentation/filtering/#full-text-match), if the query has several words, the condition will be satisfied if all of them are present in the text.


### Search by `.tags` field

Text can also be indexed when it is part of `tags`.
This is mostly useful in applications where text data can be split into groups and applications might require retrieving items based on a text search in an specific tag.

For example:

```python
from docarray import Document, DocumentArray

da = DocumentArray(
    storage='qdrant',
    config={'n_dim': 32, 'tag_indices': ['food_type', 'price']},
)
with da:
    da.extend(
        [
            Document(
                tags={
                    'food_type': 'Italian and Spanish food',
                    'price': 'cheap but not that cheap',
                },
            ),
            Document(
                tags={
                    'food_type': 'French and Italian food',
                    'price': 'on the expensive side',
                },
            ),
            Document(
                tags={
                    'food_type': 'chinese noddles',
                    'price': 'quite cheap for what you get!',
                },
            ),
        ]
    )

results_cheap = da.find('cheap', index='price')
print('searching "cheap" in <price>:\n\t', results_cheap[:, 'tags__price'])

results_italian = da.find('italian', index='food_type')
print('searching "italian" in <food_type>:\n\t', results_italian[:, 'tags__food_type'])
```

This will print:

```console
searching "cheap" in <price>:
         ['quite cheap for what you get!', 'cheap but not that cheap']
searching "italian" in <food_type>:
         ['French and Italian food', 'Italian and Spanish food']
```


```{note}
By default, if you don't specify the parameter `index` in the `find` method, the Document attribute `text` will be used for search. If you want to use a specific tags field, make sure to specify it with parameter `index`:
```python
results = da.find('cheap', index='price')
```