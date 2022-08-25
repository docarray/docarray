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
    image: redis/redis-stack:6.2.2-v5
    ports:
      - "6379:6379"
```

Then

```bash
docker-compose up
```

### Create DocumentArray with Redis backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:6379`), one can instantiate a DocumentArray with Redis storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='redis', config={'n_dim': 128})
```

The usage would be the same as the ordinary DocumentArray, but the dimension of an embedding for a Document must be provided at creation time.

**Currently, one Redis server instance is only supoorted to store one DocumentArray.** To access a DocumentArray formerly persisted, one can specify `host` and `port`.

The following example will build a DocumentArray with previously stored data on `http://localhost:6379`:

```python
from docarray import DocumentArray, Document

da = DocumentArray(
    storage='redis',
    config={'n_dim': 128},
)

da.extend([Document() for _ in range(1000)])

da2 = DocumentArray(
    storage='redis',
    config={'n_dim': 128},
)

da2.summary()
```

```text
              Documents Summary

  Length                 1000
  Homogenous Documents   True
  Common Attributes      ('id',)
  Multimodal dataclass   False

                      Attributes Summary

  Attribute   Data type      #Unique values   Has empty value
 ─────────────────────────────────────────────────────────────
  id          ('str',)       1000             False
```

To store a new DocumentArray in current Redis server, one can set `flush` to `True` so that previous DocumentArray will be cleared:

```python
from docarray import DocumentArray

da = DocumentArray(storage='redis', config={'n_dim': 128, 'flush': True})
```

Other functions behave the same as in-memory DocumentArray.


## Config

The following configs can be set:

| Name              | Description                                                                                       | Default                                                 |
|-------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `host`            | Host address of the Redis server                                                                  |                                 |
| `port`            | Port of the Redis Server                                                                          |                                                    |
| `redis_config`    | Other Redis configs in a Dict and pass to `Redis` client constructor, e.g. `socket_timeout`, `ssl`|                                                     |
| `index_name`      | Redis index name; the name of RedisSearch index to set this DocumentArray                         |                                                     |
| `n_dim`           | Dimensionality of the embeddings                                                                  |                                                     |
| `flush`           | Boolean flag indicating whether to clear previous DocumentArray in Redis                          |                                                |
| `update_schema`   | Boolean flag indicating whether to update Redis Search schema                                     |                                                 |
| `distance`        | Similarity distance metric in Redis                                                               |                                                |
| `batch_size`      | Batch size used to handle storage updates                                                         |                                                           |
| `method`          | Vector similarity index algorithm in Redis                                                        |
| `ef_construction` | Optional parameter for Redis HNSW algorithm                                                       | |
| `m`               | Optional parameter for Redis HNSW algorithm                                                       | |
| `ef_runtime`      | Optional parameter for Redis HNSW algorithm                                                       | |
| `block_size`      | Optional parameter for Redis FLAT algorithm                                                       | |
| `initial_cap`     | Optional parameter for Redis HNSW and FLAT algorithm                                              |                                                   |
| `columns`         | Other fields to stora in Document and build schema                                                |                                                   |