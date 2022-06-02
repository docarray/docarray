(weaviate)=
# Weaviate

One can use [Weaviate](https://www.semi.technology) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `weaviate-client`. You can install it via `pip install "docarray[weaviate]".` 
````

Here is a video tutorial that guides you to build a simple image search using Weaviate and Docarray.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/rBKvoIGihnY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>

## Usage

### Start Weaviate service

To use Weaviate as the storage backend, it is required to have the Weaviate service started. Create `docker-compose.yml` as follows:

```yaml
---
version: '3.4'
services:
  weaviate:
    command:
      - --host
      - 0.0.0.0
      - --port
      - '8080'
      - --scheme
      - http
    image: semitechnologies/weaviate:1.13.2
    ports:
      - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
...
```

Then

```bash
docker compose up
```

### Create DocumentArray with Weaviate backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:8080`), one can instantiate a DocumentArray with Weaviate storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='weaviate')
```

The usage would be the same as the ordinary DocumentArray.

To access a DocumentArray formerly persisted, one can specify the name, the host, the port and the protocol to connect to the server. `name` is required in this case but other connection parameters are optional. If they are not provided, then it will connect to the Weaviate service bound to `http://localhost:8080`.

Note, that the `name` parameter in `config` needs to be capitalized.

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='weaviate', config={'name': 'Persisted', 'host': 'localhost', 'port': 1234}
)

da.summary()
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Default                                            |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `host`                     | Hostname of the Weaviate server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 'localhost'                                        |
| `port`                     | port of the Weaviate server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 8080                                               |
| `protocol`                 | protocol to be used. Can be 'http' or 'https'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 'http'                                             |
| `name`                     | Weaviate class name; the class name of Weaviate object to presesent this DocumentArray                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                               |
| `serialize_config`         | [Serialization config of each Document](../../../fundamentals/document/serialization.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | None                                               |
| `distance`                 | The distance metric used to compute the distance between vectors. Must be either `cosine` or `l2-squared`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `None`, defaults to the default value in Weaviate* |
| `ef`                       | The size of the dynamic list for the nearest neighbors (used during the search). The higher ef is chosen, the more accurate, but also slower a search becomes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `None`, defaults to the default value in Weaviate* |
| `ef_construction`          | The size of the dynamic list for the nearest neighbors (used during the construction). Controls index search speed/build speed tradeoff.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `None`, defaults to the default value in Weaviate* |
| `timeout_config`           | Set the timeout configuration for all requests to the Weaviate server.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `None`, defaults to the default value in Weaviate* |
| `max_connections`          | The maximum number of connections per element in all layers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `None`, defaults to the default value in Weaviate* |
| `dynamic_ef_min`           | If using dynamic ef (set to -1), this value acts as a lower boundary. Even if the limit is small enough to suggest a lower value, ef will never drop below this value. This helps in keeping search accuracy high even when setting very low limits, such as 1, 2, or 3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `None`, defaults to the default value in Weaviate* |
| `dynamic_ef_max`           | If using dynamic ef (set to -1), this value acts as an upper boundary. Even if the limit is large enough to suggest a lower value, ef will be capped at this value. This helps to keep search speed reasonable when retrieving massive search result sets, e.g. 500+.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `None`, defaults to the default value in Weaviate* |
| `dynamic_ef_factor`        | If using dynamic ef (set to -1), this value controls how ef is determined based on the given limit. E.g. with a factor of 8, ef will be set to 8*limit as long as this value is between the lower and upper boundary. It will be capped on either end, otherwise.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `None`, defaults to the default value in Weaviate* |
| `vector_cache_max_objects` | For optimal search and import performance all previously imported vectors need to be held in memory. However, Weaviate also allows for limiting the number of vectors in memory. By default, when creating a new class, this limit is set to 2M objects. A disk lookup for a vector is orders of magnitudes slower than memory lookup, so the cache should be used sparingly.                                                                                                                                                                                                                                                                                                                                                                                                                                | `None`, defaults to the default value in Weaviate* |
| `flat_search_cutoff`       | Absolute number of objects configured as the threshold for a flat-search cutoff. If a filter on a filtered vector search matches fewer than the specified elements, the HNSW index is bypassed entirely and a flat (brute-force) search is performed instead. This can speed up queries with very restrictive filters considerably. Optional, defaults to 40000. Set to 0 to turn off flat-search cutoff entirely.                                                                                                                                                                                                                                                                                                                                                                                           | `None`, defaults to the default value in Weaviate* |
| `cleanup_interval_seconds` | How often the async process runs that “repairs” the HNSW graph after deletes and updates. (Prior to the repair/cleanup process, deleted objects are simply marked as deleted, but still a fully connected member of the HNSW graph. After the repair has run, the edges are reassigned and the datapoints deleted for good). Typically this value does not need to be adjusted, but if deletes or updates are very frequent it might make sense to adjust the value up or down. (Higher value means it runs less frequently, but cleans up more in a single batch. Lower value means it runs more frequently, but might not be as efficient with each run).                                                                                                                                                  | `None`, defaults to the default value in Weaviate* |
| `skip`                     | There are situations where it doesn’t make sense to vectorize a class. For example if the class is just meant as glue between two other class (consisting only of references) or if the class contains mostly duplicate elements (Note that importing duplicate vectors into HNSW is very expensive as the algorithm uses a check whether a candidate’s distance is higher than the worst candidate’s distance for an early exit condition. With (mostly) identical vectors, this early exit condition is never met leading to an exhaustive search on each import or query). In this case, you can skip indexing a vector all-together. To do so, set "skip" to "true". skip defaults to false; if not set to true, classes will be indexed normally. This setting is immutable after class initialization. | `None`, defaults to the default value in Weaviate* |


*You can read more about the HNSW parameters and their default values [here](https://weaviate.io/developers/weaviate/current/vector-index-plugins/hnsw.html#how-to-use-hnsw-and-parameters)

## Minimum Example

The following example shows how to use DocArray with Weaviate Document Store in order to index and search text 
Documents.

First, let's run the create the `DocumentArray` instance (make sure a Weaviate server is up and running):

```python
from docarray import DocumentArray

da = DocumentArray(
    storage="weaviate", config={"name": "Persisted", "host": "localhost", "port": 8080}
)
```

Then, we can index some Documents:

```python
from docarray import Document

da.extend(
    [
        Document(text='Persist Documents with Weaviate.'),
        Document(text='And enjoy fast nearest neighbor search.'),
        Document(text='All while using DocArray API.'),
    ]
)
```

Now, we can generate embeddings inside the database using BERT model:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def collate_fn(da):
    return tokenizer(da.texts, return_tensors='pt', truncation=True, padding=True)


da.embed(model, collate_fn=collate_fn)
```


Finally, we can query the database and print the results:

```python
results = da.find(
    DocumentArray([Document(text='How to persist Documents')]).embed(
        model, collate_fn=collate_fn
    ),
    limit=1,
)

print(results[0].text)
```

```text
Persist Documents with Weaviate.
```

## Vector search with filter

Search with `.find` can be restricted by user-defined filters. Such filters can be constructed following the guidelines 
in [Weaviate's Documentation](https://weaviate.io/developers/weaviate/current/graphql-references/filters.html).

### Example of `.find` with a filter only

Consider you store Documents with a certain tag `price` into weaviate and you want to retrieve all Documents
with `price`  lower or equal to  some `max_price` value. 


You can index such Documents as follows:
```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
da = DocumentArray(
    storage='weaviate',
    config={
        'n_dim': n_dim,
        'columns': [('price', 'float')],
    },
)

with da:
    da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])

print('\nIndexed Embeddings:\n')
for price in da[:, 'tags__price']:
    print(f'\t price={price}')
```

Then you can retrieve all documents whose price is lower than or equal to `max_price` by applying the following 
filter:

```python
max_price = 3
n_limit = 4

filter = {'path': 'price', 'operator': 'LessThanEqual', 'valueNumber': max_price}
results = da.find(filter=filter)

print('\n Returned examples that verify filter "price at most 7":\n')
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

da = DocumentArray(
    storage='weaviate',
    config={'n_dim': n_dim, 'columns': [('price', 'int')], 'distance': 'l2-squared'},
)

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )

print('\nIndexed Embeddings:\n')
for embedding, price in zip(da.embeddings, da[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

Consider we want the nearest vectors to the embedding `[8. 8. 8.]`, with the restriction that
prices must follow a filter. As an example, let's consider that retrieved documents must have `price` value lower
or equal than `max_price`. We can encode this information in weaviate using `filter = {'path': ['price'], 'operator': 'LowerThanEqual', 'valueInt': max_price}`.

Then the search with the proposed filter can implemented and used with the following code:

```python
max_price = 7
n_limit = 4

np_query = np.ones(n_dim) * 8
print(f'\nQuery vector: \t{np_query}')

filter = {'path': ['price'], 'operator': 'LessThanEqual', 'valueInt': max_price}
results = da.find(np_query, filter=filter, limit=n_limit)

print('\nEmbeddings Nearest Neighbours with "price" at most 7:\n')
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

This would print:

```bash
Embeddings Nearest Neighbours with "price" at most 7:

	embedding=[7. 7. 7.],	 price=7
	embedding=[6. 6. 6.],	 price=6
	embedding=[5. 5. 5.],	 price=5
	embedding=[4. 4. 4.],	 price=4
 ```


