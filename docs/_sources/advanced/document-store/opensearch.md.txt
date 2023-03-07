(opensearch)=

# Opensearch

You can use [Opensearch](https://opensearch.org/) as the document store for DocumentArray. It is useful when you wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `opensearch`. You can install it via `pip install "docarray[opensearch]".`
````

## Usage

### Start Opensearch service

To use Opensearch as the storage backend, it is required to have the Opensearch service started. Create `docker-compose.yml` as follows:

```yaml
version: "3.3"
services:
  opensearch:
    image: opensearchproject/opensearch:2.4.0
    environment:
      - plugins.security.disabled=true
      - discovery.type=single-node
    ports:
      - "9900:9200"
    networks:
      - os

networks:
  os:
    name: os
```

Then

```bash
pip install -U docarray[opensearch]
docker-compose up
```

### Create DocumentArray with Opensearch backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:9200`), you can instantiate a DocumentArray with Opensearch storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='opensearch', config={'n_dim': 128})
```

The usage would be the same as the ordinary DocumentArray, but the dimension of an embedding for a Document must be provided at creation time.

### Secure connection

By default, Opensearch server runs with security layer that disables the plain HTTP connection. You can pass the `host` with `api_id` or `ca_certs` inside `opensearch_config` to the constructor. For example,

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='opensearch',
    config={
        'hosts': 'https://opensearch:PRq7je_hJ4i4auh+Hq+*@localhost:9200',
        'n_dim': 128,
        'opensearch_config': {'ca_certs': '/Users/hanxiao/http_ca.crt'},
    },
)
```

Here is [the official Documentation](https://opensearch.org/docs/2.0/security-plugin/configuration/generate-certificates/) for you to get certificate, password etc.

To access a DocumentArray formerly persisted, you can specify `index_name` and the hosts.

The following example will build a DocumentArray with previously stored data from `old_stuff` on `http://localhost:9200`:

```python
from docarray import DocumentArray, Document

da = DocumentArray(
    storage='opensearch',
    config={'index_name': 'old_stuff', 'n_dim': 128},
)

with da:
    da.extend([Document() for _ in range(1000)])

da2 = DocumentArray(
    storage='opensearch',
    config={'index_name': 'old_stuff', 'n_dim': 128},
)

da2.summary()
```

```text
              Documents Summary

  Length                 2000
  Homogenous Documents   True
  Common Attributes      ('id', 'embedding')

                      Attributes Summary

  Attribute   Data type      #Unique values   Has empty value
 ─────────────────────────────────────────────────────────────
  embedding   ('ndarray',)   1000             False
  id          ('str',)       1000             False

              Storage Summary

  Backend            OpenSearch
  Host               http://localhost:9200
  Distance           cosine
  Vector dimension   128
  ES config          {}

[0.14890289 0.3168339  0.03050802 0.06785086 0.94719299 0.32490566
 ...]
```

Other functions behave the same as in-memory DocumentArray.

### Bulk request customization

You can customize how bulk requests are sent to Opensearch when adding Documents by adding additional `kwargs` on `extend` method call. See [the official documentation](https://opensearch.org/docs/1.2/opensearch/rest-api/document-apis/bulk/) for more details. See the following code for example:

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3

da = DocumentArray(
    storage='opensearch',
    config={'n_dim': 3, 'columns': {'price': 'int'}, 'distance': 'l2'},
)

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ],
        thread_count=4,
        chunk_size=500,
        max_chunk_bytes=104857600,
        queue_size=4,
    )
```

````{admonition} Note
:class: note
`batch_size` configuration will be overriden by `chunk_size` kwargs if provided
````

```{tip}
You can read more about parallel bulk config and their default values [here](https://github.com/opensearch-project/opensearch-py/blob/main/opensearchpy/helpers/actions.py)
```

### Vector search with filter query

You can perform Approximate Nearest Neighbor Search and pre-filter results using a filter query .

Consider Documents with embeddings `[0,0,0]` up to `[9,9,9]` where the Document with embedding `[i,i,i]`
has as tag `price` with value `i`. We can create such example with the following code:

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3

da = DocumentArray(
    storage='opensearch',
    config={'n_dim': n_dim, 'columns': {'price': 'int'}, 'distance': 'l2'},
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

Consider we want the nearest vectors to the embedding `[8. 8. 8.]`, with the restriction that
prices must follow a filter. As an example, let's consider that retrieved Documents must have `price` value lower
or equal than `max_price`. We can encode this information in OpenSearch using `filter = {'range': {'price': {'lte': max_price}}}`.

Then the search with the proposed filter can be implemented and used with the following code:

```python
max_price = 7
n_limit = 4

np_query = np.ones(n_dim) * 8
print(f'\nQuery vector: \t{np_query}')

filter = {'range': {'price': {'lte': max_price}}}
results = da.find(np_query, filter=filter, limit=n_limit)

print('\nEmbeddings Nearest Neighbours with "price" at most 7:\n')
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

This would print:

```bash
Embeddings Nearest Neighbours with "price" at most 7:

 embedding=[7. 7. 7.],  price=7
 embedding=[6. 6. 6.],  price=6
 embedding=[5. 5. 5.],  price=5
 embedding=[4. 4. 4.],  price=4
 ```

Additionally you can tune the approximate kNN for speed or accuracy by providing `num_candidates` kwarg when calling the `find` method:

```python
results = da.find(np_query, filter=filter, limit=n_limit, num_candidates=100)
```

```{tip}
You can read more about approximate kNN tuning [here](https://opensearch.org/docs/latest/search-plugins/knn/index/)
```

### Search by filter query

You can search with user-defined query filters using the `.find` method. Such queries can be constructed following the
guidelines in [OpenSearch's Documentation](https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/).

Consider you store Documents with a certain tag `price` into OpenSearch and you want to retrieve all Documents
with `price`  lower or equal to  some `max_price` value.

You can index such Documents as follows:

```python
from docarray import Document, DocumentArray

n_dim = 3
da = DocumentArray(
    storage='opensearch',
    config={
        'n_dim': n_dim,
        'columns': {'price': 'float'},
    },
)

with da:
    da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])

print('\nIndexed Prices:\n')
for price in da[:, 'tags__price']:
    print(f'\t price={price}')
```

Then you can retrieve all Documents whose price is lower than or equal to `max_price` by applying the following
filter:

```python
max_price = 3
n_limit = 4

filter = {
    'range': {
        'price': {
            'lte': max_price,
        }
    }
}
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

### Search by `.text` field

Text search can be easily leveraged in a `DocumentArray` with `storage='opensearch'`.
To do this text needs to be indexed using the boolean flag `'index_text'` which is set when
the `DocumentArray` is created  with `config={'index_text': True, ...}`.
The following example builds a `DocumentArray` with several Documents containing text and searches
for those that have `pizza` in their text description.

```python
from docarray import DocumentArray, Document

da = DocumentArray(storage='opensearch', config={'n_dim': 2, 'index_text': True})
with da:
    da.extend(
        [
            Document(text='Person eating'),
            Document(text='Person eating pizza'),
            Document(text='Pizza restaurant'),
        ]
    )

pizza_docs = da.find('pizza')
pizza_docs[:, 'text']
```

will print

```text
['Pizza restaurant', 'Person eating pizza']
```

### Search by `.tags` field

Text can also be indexed when it is part of `tags`.
This is mostly useful in applications where text data can be split into groups and applications might require
retrieving items based on a text search in an specific tag.

For example:

```python
from docarray import DocumentArray, Document

da = DocumentArray(
    storage='opensearch', config={'n_dim': 32, 'tag_indices': ['food_type', 'price']}
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

will print

```text
searching "cheap" in <price>:
  ['cheap but not that cheap', 'quite cheap for what you get!']
searching "italian" in <food_type>:
  ['Italian and Spanish food', 'French and Italian food']
```

````{admonition} Note
:class: note
By default, if you don't specify the parameter `index` in the `find` method, the Document attribute `text` will be used
for search. If you want to use a specific tags field, make sure to specify it with parameter `index`:
```python
results = da.find('cheap', index='price')
```
````

## Configuration

| Name                | Description                                                                                                                            | Default                                              |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `hosts`             | Hostname of the Opensearch server                                                                                                      | `http://localhost:9200`                              |
| `opensearch_config` | Other OpenSearch configs in a Dict and pass to `Opensearch` client constructor, e.g. `cloud_id`, `api_key`                             | None                                                 |
| `index_name`        | Opensearch index name; the class name of Opensearch index object to set this DocumentArray                                             | None                                                 |
| `n_dim`             | Dimensionality of the embeddings                                                                                                       | None                                                 |
| `distance`          | Similarity metric in Opensearch                                                                                                        | `cosinesimil`                                        |
| `ef_construction`   | The size of the dynamic list for the nearest neighbors.                                                                                | `None`, defaults to the default value in OpenSearch* |
| `m`                 | Similarity metric in Opensearch                                                                                                        | `None`, defaults to the default value OpenSearch*    |
| `index_text`        | Boolean flag indicating whether to index `.text` or not                                                                                | False                                                |
| `tag_indices`       | List of tags to index                                                                                                                  | False                                                |
| `batch_size`        | Batch size used to handle storage refreshes/updates                                                                                    | 64                                                   |
| `list_like`         | Controls if ordering of Documents is persisted in the Database. Disabling this breaks list-like features, but can improve performance. | True                                                 |

```{tip}
You can read more about HNSW parameters and their default values [here](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/)
```

```{tip}
Note that it is plural `hosts` not `host`, to comply with Opensearch client's interface.
```
