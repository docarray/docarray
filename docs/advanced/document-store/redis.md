(redis)=
# Redis

You can use [Redis](https://redis.io) as the document store for DocumentArray. It is useful when you want to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

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
pip install -U docarray[redis]
docker-compose up
```

### Create DocumentArray with Redis backend

Assuming the service is started using the default configuration (i.e. server address is `localhost:6379`), you can instantiate a DocumentArray with Redis storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='redis', config={'host': 'localhost', 'port': 6379, 'n_dim': 128}
)
```

The usage will be the same as the ordinary DocumentArray, but the dimension of an embedding for a Document must be provided at creation time.

To access a previously stored DocumentArray, you can specify `index_name` and set `host` and `port` to match with the previuosly stored DocumentArray.

The following example builds a DocumentArray from previously stored data on `localhost:6379`:

```python
from docarray import DocumentArray, Document

with DocumentArray(
    storage='redis',
    config={
        'n_dim': 128,
        'index_name': 'idx',
    },
) as da:
    da.extend([Document() for _ in range(1000)])

da2 = DocumentArray(
    storage='redis',
    config={
        'n_dim': 128,
        'index_name': 'idx',
    },
)

da2.summary()
```

```{dropdown} Output
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
│   update_schema     True        │
│   distance          COSINE      │
│   redis_config      {}          │
│   index_text        False       │
│   tag_indices       []          │
│   batch_size        64          │
│   method            HNSW        │
│   ef_construction   200         │
│   m                 16          │
│   ef_runtime        10          │
│   block_size        1048576     │
│   initial_cap       None        │
│   columns           {}          │
│                                 │
╰─────────────────────────────────╯
```

Other functions behave the same as in-memory DocumentArray.


### Vector search with filter query

You can perform Vector Similarity Search based on [FLAT or HNSW algorithm](vector-search-index) and pre-filter results using a filter query that is based on [MongoDB's Query](https://www.mongodb.com/docs/manual/reference/operator/query/). The following tags filters can be combine with `$and` and `$or`:

- `$eq` - Equal to (number, string)
- `$ne` - Not equal to (number, string)
- `$gt` - Greater than (number)
- `$gte` - Greater than or equal to (number)
- `$lt` - Less than (number)
- `$lte` - Less than or equal to (number)


Consider Documents with embeddings `[0, 0, 0]` up to `[9, 9, 9]` where the Document with embedding `[i, i, i]`
has tag `price` with a number value, tag `color` with a string value and tag `stock` with a boolean value. You can create such example with the following code:

```python
import numpy as np
from docarray import Document, DocumentArray

n_dim = 3

da = DocumentArray(
    storage='redis',
    config={
        'n_dim': n_dim,
        'columns': {'price': 'int', 'color': 'str', 'stock': 'bool'},
        'distance': 'L2',
    },
)

da.extend(
    [
        Document(
            id=f'{i}',
            embedding=i * np.ones(n_dim),
            tags={'price': i, 'color': 'blue', 'stock': i % 2 == 0},
        )
        for i in range(10)
    ]
)
da.extend(
    [
        Document(
            id=f'{i+10}',
            embedding=i * np.ones(n_dim),
            tags={'price': i, 'color': 'red', 'stock': i % 2 == 0},
        )
        for i in range(10)
    ]
)

print('\nIndexed price, color and stock:\n')
for doc in da:
    print(
        f"\tembedding={doc.embedding},\t color={doc.tags['color']},\t stock={doc.tags['stock']}"
    )
```

Consider the case where you want the nearest vectors to the embedding `[8.,  8.,  8.]`, with the restriction that prices, colors and stock must pass a filter. For example, let's consider that retrieved Documents must have a `price` value lower than or equal to `max_price`, have `color` equal to `blue` and have `stock` equal to `True`. We can encode this information in Redis using

```text
{
    "price": {"$lte": max_price},
    "color": {"$gt": color},
    "stock": {"$eq": True},
}
```
or 

```text
{
    "$and": {
        "price": {"$lte": max_price},
        "color": {"$gt": color},
        "stock": {"$eq": True},
    }
}
```

Then the search with the proposed filter can be used as follows:
```python
max_price = 7
color = "blue"
n_limit = 5

np_query = np.ones(n_dim) * 8
print(f'\nQuery vector: \t{np_query}')

filter = {
    "price": {"$lte": max_price},
    "color": {"$eq": color},
    "stock": {"$eq": True},
}

results = da.find(np_query, filter=filter, limit=n_limit)

print(
    '\nEmbeddings Approximate Nearest Neighbours with "price" at most 7, "color" blue and "stock" False:\n'
)
for doc in results:
    print(
        f" score={doc.scores['score'].value},\t embedding={doc.embedding},\t price={doc.tags['price']},\t color={doc.tags['color']},\t stock={doc.tags['stock']}"
    )
```

This will print:

```console
Embeddings Approximate Nearest Neighbours with "price" at most 7, "color" blue and "stock" True:

 score=12,	 embedding=[6. 6. 6.],	 price=6,	 color=blue,	 stock=True
 score=48,	 embedding=[4. 4. 4.],	 price=4,	 color=blue,	 stock=True
 score=108,	 embedding=[2. 2. 2.],	 price=2,	 color=blue,	 stock=True
 score=192,	 embedding=[0. 0. 0.],	 price=0,	 color=blue,	 stock=True
```
More example filter expresses
- A Nike shoes or price less than `100`

```JSON
{
    "$or": {
        "brand": {"$eq": "Nike"},
        "price": {"$lt": 100}
    }
}
```

- A Nike shoes **and** either price is less than `100` or color is `"blue"`

```JSON
{
    "brand": {"$eq": "Nike"},
    "$or": {
        "price": {"$lt": 100},
        "color": {"$eq": "blue"},
    },
}
```

- A Nike shoes **or** both price is less than `100` and color is `"blue"`

```JSON
{
    "$or": {
        "brand": {"$eq": "Nike"},
        "$and": {
            "price": {"$lt": 100},
            "color": {"$eq": "blue"},
        },
    }
}
```


(vector-search-index)=
### Update Vector Search Indexing Schema

Redis vector similarity supports two indexing methods:

- **FLAT**: Brute-force search. 
- **HNSW**: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.

Both methods have some mandatory parameters and optional parameters.

```{tip}
Read more about HNSW or FLAT parameters and their default values [here](https://redis.io/docs/stack/search/reference/vectors/#querying-vector-fields).
```

You can update the search indexing schema on an existing DocumentArray by setting `update_schema` to `True` and changing your configuratoin parameters.

Consider you store Documents with default indexing method `'HNSW'` and distance `'L2'`, and want to find the nearest vectors to the embedding `[8. 8. 8.]`:

```python
import numpy as np
from docarray import Document, DocumentArray

n_dim = 3

da = DocumentArray(
    storage='redis',
    config={
        'n_dim': n_dim,
        'index_name': 'idx',
        'distance': 'L2',
    },
)

da.extend([Document(id=f'{i}', embedding=i * np.ones(n_dim)) for i in range(10)])

np_query = np.ones(n_dim) * 8
n_limit = 5

results = da.find(np_query, limit=n_limit)

print('\nEmbeddings Approximate Nearest Neighbours:\n')
for doc in results:
    print(f" embedding={doc.embedding},\t score={doc.scores['score'].value}")
```

This will print:

```console
Embeddings Approximate Nearest Neighbours:

 embedding=[8. 8. 8.],   score=0
 embedding=[7. 7. 7.],   score=3
 embedding=[9. 9. 9.],   score=3
 embedding=[6. 6. 6.],   score=12
 embedding=[5. 5. 5.],   score=27
```

Then you can use a different search indexing schema on the current DocumentArray as follows:
```python
da2 = DocumentArray(
    storage='redis',
    config={
        'n_dim': n_dim,
        'index_name': 'idx',
        'update_schema': True,
        'distance': 'COSINE',
    },
)

results = da.find(np_query, limit=n_limit)

print('\nEmbeddings Approximate Nearest Neighbours:\n')
for doc in results:
    print(f" embedding={doc.embedding},\t score={doc.scores['score'].value}")
```

This will print:

```console
Embeddings Approximate Nearest Neighbours:

 embedding=[3. 3. 3.],	 score=0
 embedding=[6. 6. 6.],	 score=0
 embedding=[4. 4. 4.],	 score=5.96046447754e-08
 embedding=[1. 1. 1.],	 score=5.96046447754e-08
 embedding=[8. 8. 8.],	 score=5.96046447754e-08
```


### Search by `.text` field

You can perform full-text search in a `DocumentArray` with `storage='redis'`. 
To do this, text needs to be indexed using the boolean flag `'index_text'` which is set when the `DocumentArray` is created  with `config={'index_text': True, ...}`.
The following example builds a `DocumentArray` with several documents containing text and searches for those that have `token1` in their text description.

```python
from docarray import Document, DocumentArray

da = DocumentArray(storage='redis', config={'n_dim': 2, 'index_text': True})
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

The default similarity ranking algorithm is `BM25`. Besides, `TFIDF`, `TFIDF.DOCNORM`, `DISMAX`, `DOCSCORE` and `HAMMING` are also supported by [RediSearch](https://redis.io/docs/stack/search/reference/scoring/). You can change it by specifying `scorer` in function `find`:

```python
results = da.find('token1 token3', scorer='TFIDF.DOCNORM')
print('scorer=TFIDF.DOCNORM:')
print(results[:, 'text'])

results = da.find('token1 token3')
print('scorer=BM25:')
print(results[:, 'text'])
```

This will print:

```console
scorer=TFIDF.DOCNORM:
['token1 token2', 'token1 token2 token3', 'token2 token3 token4']
scorer=BM25:
['token1 token2 token3', 'token1 token2', 'token2 token3 token4']
```

### Search by `.tags` field

Text can also be indexed when it is part of `tags`.
This is mostly useful in applications where text data can be split into groups and applications might require retrieving items based on a text search in an specific tag.

For example:

```python
from docarray import Document, DocumentArray

da = DocumentArray(
    storage='redis',
    config={'n_dim': 32, 'tag_indices': ['food_type', 'price']},
)
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
	 ['cheap but not that cheap', 'quite cheap for what you get!']
searching "italian" in <food_type>:
	 ['Italian and Spanish food', 'French and Italian food']
```

```{note}
By default, if you don't specify the parameter `index` in the `find` method, the Document attribute `text` will be used for search. If you want to use a specific tags field, make sure to specify it with parameter `index`:
```python
results = da.find('cheap', index='price')
```



## Configuration

The following configs can be set:

| Name              | Description                                                                                       | Default                                           |
|-------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------- |
| `host`            | Host address of the Redis server                                                                  | `'localhost'`                                     |
| `port`            | Port of the Redis Server                                                                          | `6379`                                            |
| `redis_config`    | Other Redis configs in a Dict and pass to `Redis` client constructor, e.g. `socket_timeout`, `ssl`| `{}`                                              |
| `index_name`      | Redis index name; the name of RedisSearch index to set this DocumentArray                         | `None`                                            |
| `n_dim`           | Dimensionality of the embeddings                                                                  | `None`                                            |
| `update_schema`   | Boolean flag indicating whether to update Redis Search schema                                     | `True`                                            |
| `distance`        | Similarity distance metric in Redis, one of {`'L2'`, `'IP'`, `'COSINE'`}                          | `'COSINE'`                                        |
| `batch_size`      | Batch size used to handle storage updates                                                         | `64`                                              |
| `method`          | Vector similarity index algorithm in Redis, either `FLAT` or `HNSW`                               | `'HNSW'`                                          |
| `index_text`      | Boolean flag indicating whether to index `.text`. `True` will enable full text search on `.text`  | `None`                                            |
| `tag_indices`     | List of tags to index as text field                                                               | `[]`                                              |
| `ef_construction` | Optional parameter for Redis HNSW algorithm                                                       | `200`                                             |
| `m`               | Optional parameter for Redis HNSW algorithm                                                       | `16`                                              |
| `ef_runtime`      | Optional parameter for Redis HNSW algorithm                                                       | `10`                                              |
| `block_size`      | Optional parameter for Redis FLAT algorithm                                                       | `1048576`                                         |
| `initial_cap`     | Optional parameter for Redis HNSW and FLAT algorithm                                              | `None`, defaults to the default value in Redis    |
| `columns`         | Other fields to store in Document and build schema                                                | `None`                                            |

You can check the default values in [the docarray source code](https://github.com/jina-ai/docarray/blob/main/docarray/array/storage/redis/backend.py)
For vector search configurations, default values are those of the database backend, you can find them in [redis documentation](https://redis.io/docs/stack/search/reference/vectors/)

```{note}
We will support geo-filtering soon. 
The benchmark test is on the way.
```
