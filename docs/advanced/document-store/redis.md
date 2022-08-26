(redis)=
# Redis

One can use [Redis](https://redis.io) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `redis`. You can install it via `pip install "docarray[redis]".`
````

## Usage

### Start Redis service

To use Redis as the storage backend, it is required to have the Redis service started. Create `docker-compose.yml` as follows:

```yaml
version: "3.3"
services:
  redis:
    image: redislabs/redisearch:2.6.0
    ports:
      - "6379:6379"
```

Then

```bash
docker-compose up
```

### Create DocumentArray with Redis backend

Assuming service is started using the default configuration (i.e. server address is `localhost:6379`), one can instantiate a DocumentArray with Redis storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='redis', config={'n_dim': 128})
```

The usage would be the same as the ordinary DocumentArray, but the dimension of an embedding for a Document must be provided at creation time.

```{caution}
Currently, one Redis server instance can only support to store a single DocumentArray.
```

To store a new DocumentArray in current Redis server, one can set `flush` to `True` so that previous DocumentArray will be cleared:

```python
from docarray import DocumentArray

da = DocumentArray(storage='redis', config={'n_dim': 128, 'flush': True})
```

To access a previously stored DocumentArray, one can specify `host` and `port`.

The following example will build a DocumentArray with previously stored data on `localhost:6379`:

```python
from docarray import DocumentArray, Document

with DocumentArray(
    storage='redis',
    config={'n_dim': 128, 'flush': True},
) as da:
    da.extend([Document() for _ in range(1000)])

da2 = DocumentArray(
    storage='redis',
    config={'n_dim': 128},
)

da2.summary()
```

```console
╭────────────── Documents Summary ──────────────╮
│                                               │
│   Type                   DocumentArrayRedis   │
│   Length                 1000                 │
│   Homogenous Documents   True                 │
│   Common Attributes      ('id',)              │
│   Multimodal dataclass   False                │
│                                               │
╰───────────────────────────────────────────────╯
╭───────────────────── Attributes Summary ─────────────────────╮
│                                                              │
│   Attribute   Data type   #Unique values   Has empty value   │
│  ──────────────────────────────────────────────────────────  │
│   id          ('str',)    1000             False             │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
╭─── DocumentArrayRedis Config ───╮
│                                 │
│   n_dim             128         │
│   host              localhost   │
│   port              6379        │
│   index_name        idx         │
│   flush             False       │
│   update_schema     True        │
│   distance          COSINE      │
│   redis_config      {}          │
│   batch_size        64          │
│   method            HNSW        │
│   ef_construction   200         │
│   m                 16          │
│   ef_runtime        10          │
│   block_size        1048576     │
│   initial_cap       None        │
│   columns           []          │
│                                 │
╰─────────────────────────────────╯
```



Other functions behave the same as in-memory DocumentArray.


### Vector search with filter query

One can perform Vector Similarity Search based on FLAT or HNSW algorithm and pre-filter results using a filter query that is based on [MongoDB's Query](https://www.mongodb.com/docs/manual/reference/operator/query/). We currently support a subset of those selectors:

- `$eq` - Equal to (number, string)
- `$ne` - Not equal to (number, string)
- `$gt` - Greater than (number)
- `$gte` - Greater than or equal to (number)
- `$lt` - Less than (number)
- `$lte` - Less than or equal to (number)


Consider Documents with embeddings `[0,0,0]` up to `[9,9,9]` where the document with embedding `[i,i,i]`
has tag `price` with number value and tag `color` with string value. We can create such example with the following code:

```python
import numpy as np
from docarray import Document, DocumentArray

n_dim = 3

da = DocumentArray(
    storage='redis',
    config={
        'n_dim': n_dim,
        'columns': [('price', 'int'), ('color', 'str')],
        'flush': True,
    },
)

da.extend(
    [
        Document(
            id=f'{i}', embedding=i * np.ones(n_dim), tags={'price': i, 'color': 'red'}
        )
        for i in range(10)
    ]
)
da.extend(
    [
        Document(
            id=f'{i+10}',
            embedding=i * np.ones(n_dim),
            tags={'price': i, 'color': 'blue'},
        )
        for i in range(10)
    ]
)

print('\nIndexed prices and colors:\n')
for embedding, price, color in zip(
    da.embeddings, da[:, 'tags__price'], da[:, 'tags__color']
):
    print(f'\tembedding={embedding},\t price={price},\t color={color}')
```

Consider we want the nearest vectors to the embedding `[8. 8. 8.]`, with the restriction that
prices and color must follow a filter. For example, let's consider that retrieved documents must have a `price` value lower than or equal to `max_price` and have `color` equal to `color`. We can encode this information in Redis using `{'price': {'$lte': max_price}, 'color': {'$eq': color}}`.

Then the search with the proposed filter can be used as follows:
```python
max_price = 7
color = 'red'
n_limit = 5

np_query = np.ones(n_dim) * 8
print(f'\nQuery vector: \t{np_query}')

filter = {'price': {'$lte': max_price}, 'color': {'$eq': color}}
results = da.find(np_query, filter=filter, limit=n_limit)

print(
    '\nEmbeddings Approximate Nearest Neighbours with "price" at most 7 and "color" red:\n'
)
for embedding, price, color, score in zip(
    results.embeddings,
    results[:, 'tags__price'],
    results[:, 'tags__color'],
    results[:, 'scores'],
):
    print(
        f' score={score["score"].value},\t embedding={embedding},\t price={price},\t color={color}'
    )
```

This would print:

```console
Embeddings Approximate Nearest Neighbours with "price" at most 7 and "color" red:

 embedding=[3. 3. 3.],   price=3,        color=red,      score=0
 embedding=[6. 6. 6.],   price=6,        color=red,      score=0
 embedding=[1. 1. 1.],   price=1,        color=red,      score=5.96046447754e-08
 embedding=[2. 2. 2.],   price=2,        color=red,      score=5.96046447754e-08
 embedding=[4. 4. 4.],   price=4,        color=red,      score=5.96046447754e-08
```

### Update Vector Search Indexing Schema

Redis vector similarity supports two indexing methods:

- FLAT - Brute-force search. 

- HNSW - Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.

Both methods have some mandatory parameters and optional parameters.

```{tip}
Read more about HNSW or FLAT parameters and their default values [here](https://redis.io/docs/stack/search/reference/vectors/#querying-vector-fields).
```

You can update the search indexing schema on existing DocumentArray by simply set `update_schema` to `True` and change your config parameters.

Consider you store Documents with default indexing method `'HNSW'` and distance `'COSINE'`, and find the nearest vectors to the embedding `[8. 8. 8.]`.

```python
import numpy as np
from docarray import Document, DocumentArray

n_dim = 3

da = DocumentArray(
    storage='redis',
    config={
        'n_dim': n_dim,
        'flush': True,
    },
)

da.extend([Document(id=f'{i}', embedding=i * np.ones(n_dim)) for i in range(10)])

np_query = np.ones(n_dim) * 8
n_limit = 5

results = da.find(np_query, limit=n_limit)

print('\nEmbeddings Approximate Nearest Neighbours:\n')
for embedding, score in zip(
    results.embeddings,
    results[:, 'scores'],
):
    print(f' embedding={embedding},\t score={score["score"].value}')
```

This would print:

```console
Embeddings Approximate Nearest Neighbours:

 embedding=[3. 3. 3.],   score=0
 embedding=[6. 6. 6.],   score=0
 embedding=[1. 1. 1.],   score=5.96046447754e-08
 embedding=[2. 2. 2.],   score=5.96046447754e-08
 embedding=[4. 4. 4.],   score=5.96046447754e-08
```

Then you can use a different search indexing schema on current DocumentArray as follows:
```python
da2 = DocumentArray(
    storage='redis',
    config={
        'n_dim': n_dim,
        'update_schema': True,
        'distance': 'L2',
    },
)

results = da.find(np_query, limit=n_limit)

print('\nEmbeddings Approximate Nearest Neighbours:\n')
for embedding, score in zip(
    results.embeddings,
    results[:, 'scores'],
):
    print(f' embedding={embedding},\t score={score["score"].value}')
```

This would print:

```console
Embeddings Approximate Nearest Neighbours:

 embedding=[8. 8. 8.],   score=0
 embedding=[9. 9. 9.],   score=3
 embedding=[7. 7. 7.],   score=3
 embedding=[6. 6. 6.],   score=12
 embedding=[5. 5. 5.],   score=27
```


## Config

The following configs can be set:

| Name              | Description                                                                                       | Default                                           |
|-------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------- |
| `host`            | Host address of the Redis server                                                                  | `'localhost'`                                     |
| `port`            | Port of the Redis Server                                                                          | `6379`                                            |
| `redis_config`    | Other Redis configs in a Dict and pass to `Redis` client constructor, e.g. `socket_timeout`, `ssl`| `{}`                                              |
| `index_name`      | Redis index name; the name of RedisSearch index to set this DocumentArray                         | `'idx'`                                           |
| `n_dim`           | Dimensionality of the embeddings                                                                  | `None`                                            |
| `flush`           | Boolean flag indicating whether to clear previous DocumentArray in Redis                          | `False`                                           |
| `update_schema`   | Boolean flag indicating whether to update Redis Search schema                                     | `True`                                            |
| `distance`        | Similarity distance metric in Redis                                                               | `'COSINE'`                                        |
| `batch_size`      | Batch size used to handle storage updates                                                         | `64`                                              |
| `method`          | Vector similarity index algorithm in Redis, either `FLAT` or `HNSW`                               | `'HNSW'`                                          |
| `ef_construction` | Optional parameter for Redis HNSW algorithm                                                       | `200`                                             |
| `m`               | Optional parameter for Redis HNSW algorithm                                                       | `16`                                              |
| `ef_runtime`      | Optional parameter for Redis HNSW algorithm                                                       | `10`                                              |
| `block_size`      | Optional parameter for Redis FLAT algorithm                                                       | `1048576`                                         |
| `initial_cap`     | Optional parameter for Redis HNSW and FLAT algorithm                                              | `None`, defaults to the default value in Redis    |
| `columns`         | Other fields to store in Document and build schema                                                | `None`                                            |

You can check the default values in [the docarray source code](https://github.com/jina-ai/docarray/blob/main/docarray/array/storage/redis/backend.py)


```{note}
The Redis storage backend will support storing multiple DocumentArrays, full-text search, more query conitions and geo-filtering soon.
The benchmark test is on the way.
```
