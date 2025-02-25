# Elasticsearch Document Index

DocArray comes with two Document Indexes for [Elasticsearch](https://www.elastic.co/elasticsearch/):

- [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex], based on [Elasticsearch 8](https://github.com/elastic/elasticsearch).
- [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex], based on [Elasticsearch 7.10](https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-0).

!!! tip "Should you use ES v7 or v8?"
    [Elasticsearch v8](https://www.elastic.co/blog/whats-new-elastic-8-0-0) is the current version of ES and offers
    **native vector search (ANN) support**, alongside text and range search.

    [Elasticsearch v7.10](https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-0) can store vectors, but
    **does _not_ support native ANN vector search**, but only exhaustive (i.e. slow) vector search, alongside text and range search.

    Some users prefer to use ES v7.10 because it is available under a [different license](https://www.elastic.co/pricing/faq/licensing) to ES v8.0.0.

!!! note "Installation"
    To use [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex], you need to install the following dependencies:

    ```console
    pip install elasticsearch==8.6.2
    pip install elastic-transport
    ```

    To use [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex], you need to install the following dependencies:
    
    ```console
    pip install elasticsearch==7.10.1
    pip install elastic-transport
    ```


The following example is based on [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex],
but will also work for [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex].


## Basic usage
This snippet demonstrates the basic usage of [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex]. It defines a document schema with a title and an embedding, 
creates ten dummy documents with random embeddings, initializes an instance of [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex] to index these documents, 
and performs a vector similarity search to retrieve the ten most similar documents to a given query vector.

```python
from docarray import BaseDoc, DocList
from docarray.index import ElasticDocIndex  # or ElasticV7DocIndex
from docarray.typing import NdArray
import numpy as np


# Define the document schema.
class MyDoc(BaseDoc):
    title: str
    embedding: NdArray[128]


# Create dummy documents.
docs = DocList[MyDoc](
    MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10)
)

# Initialize a new ElasticDocIndex instance and add the documents to the index.
doc_index = ElasticDocIndex[MyDoc](index_name='my_index')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, search_field='embedding', limit=10)
```



## Initialize


You can use docker compose to create a local Elasticsearch service with the following `docker-compose.yml`.

```yaml
version: "3.3"
services:
  elastic:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.6.2
    environment:
      - xpack.security.enabled=false
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xmx1024m
    ports:
      - "9200:9200"
    networks:
      - elastic

networks:
  elastic:
    name: elastic
```

Run the following command in the folder of the above `docker-compose.yml` to start the service:

```bash
docker compose up
```

### Schema definition

To construct an index, you first need to define a schema in the form of a `Document`.

There are a number of configurations you can pack into your schema:

- Every field in your schema will become one column in the database
- For vector fields, such as `NdArray`, `TorchTensor`, or `TensorflowTensor`, you need to specify a dimensionality to be able to perform vector search
- You can override the default column type for every field by passing any [ES field data type](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/mapping-types.html) to `field_name: Type = Field(col_type=...)`. You can see an example of this in the [section on keyword filters](#keyword-filter).

Additionally, you can pass a `hosts` argument to the `__init__()` method to connect to an ES instance.
By default, it is `http://localhost:9200`. 

```python
import numpy as np
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    # specify tensor field with dimensionality 128
    tensor: NdArray[128]
    # alternative and equivalent definition:
    # tensor: NdArray = Field(dims=128)


doc_index = ElasticDocIndex[SimpleDoc](hosts='http://localhost:9200')
```

## Index

Now that you have a Document Index, you can add data to it, using the [`index()`][docarray.index.abstract.BaseDocIndex.index] method.
The `.num_docs()` method returns the total number of documents in the index.

```python
from docarray import DocList

# create some random data
docs = DocList[SimpleDoc]([SimpleDoc(tensor=np.ones(128)) for _ in range(64)])

doc_index.index(docs)

print(f'number of docs in the index: {doc_index.num_docs()}')
```

As you can see, `DocList[SimpleDoc]` and `ElasticDocIndex[SimpleDoc]` both have `SimpleDoc` as a parameter.
This means that they share the same schema, and in general, both the Document Index and the data that you want to store need to have compatible schemas.

!!! question "When are two schemas compatible?"
    The schemas of your Document Index and data need to be compatible with each other.
    
    Let's say A is the schema of your Document Index and B is the schema of your data.
    There are a few rules that determine if schema A is compatible with schema B.
    If _any_ of the following are true, then A and B are compatible:

    - A and B are the same class
    - A and B have the same field names and field types
    - A and B have the same field names, and, for every field, the type of B is a subclass of the type of A

    In particular, this means that you can easily [index predefined documents](#using-a-predefined-document-as-schema) into a Document Index.



## Vector search

Now that you have indexed your data, you can perform vector similarity search using the [`find()`][docarray.index.abstract.BaseDocIndex.find] method.

You can use the [`find()`][docarray.index.abstract.BaseDocIndex.find] function with a document of the type `MyDoc` 
to find similar documents within the Document Index:

You can use the `limit` argument to configure how many documents to return.

!!! note
    [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex] uses Elasticsearch v7.10.1, which does not support approximate nearest neighbour algorithms such as HNSW.
    This can lead to poor performance when the search involves many vectors.
    [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex] does not have this limitation.

=== "Search by Document"

    ```python
    # create a query document
    query = SimpleDoc(tensor=np.ones(128))

    # find similar documents
    matches, scores = doc_index.find(query, search_field='tensor', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(128)

    # find similar documents
    matches, scores = doc_index.find(query, search_field='tensor', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

To peform a vector search, you need to specify a `search_field`. This is the field that serves as the
basis of comparison between your query and the documents in the Document Index.

In this example you only have one field (`tensor`) that is a vector, so you can trivially choose that one.
In general, you could have multiple fields of type `NdArray` or `TorchTensor` or `TensorFlowTensor`, and you can choose
which one to use for the search.

The [`find()`][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on the subindex level, you can use the [`find_subindex()`][docarray.index.abstract.BaseDocIndex.find_subindex] method, which returns a named tuple containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#configuration).


### Batched search

You can also search for multiple documents at once, in a batch, using the [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method.

=== "Search by Documents"

    ```python
    # create some query Documents
    queries = DocList[SimpleDoc](SimpleDoc(tensor=np.random.rand(128)) for i in range(3))

    # find similar documents
    matches, scores = doc_index.find_batched(queries, search_field='tensor', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 128)

    # find similar documents
    matches, scores = doc_index.find_batched(query, search_field='tensor', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.



## Filter

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding filter query. 
The query should follow [Elastic's query language](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html).

The `filter()` method accepts queries that follow the [Elasticsearch Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html) and consists of leaf and compound clauses.

Using this, you can perform [keyword filters](#keyword-filter), [geolocation filters](#geolocation-filter) and [range filters](#range-filter).

### Keyword filter

To filter documents in your index by keyword, you can use `Field(col_type='keyword')` to enable keyword search for given fields:

```python
class NewsDoc(BaseDoc):
    text: str
    category: str = Field(col_type='keyword')  # enable keyword filtering


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(id='0', text='this is a news for sport', category='sport'),
    NewsDoc(id='1', text='this is a news for finance', category='finance'),
    NewsDoc(id='2', text='this is another news for sport', category='sport'),
]
doc_index.index(index_docs)

# search with filer
query_filter = {'terms': {'category': ['sport']}}
docs = doc_index.filter(query_filter)
```

### Geolocation filter

To filter documents in your index by geolocation, you can use `Field(col_type='geo_point')` on a given field:

```python
class NewsDoc(BaseDoc):
    text: str
    location: dict = Field(col_type='geo_point')  # enable geolocation filtering


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(text='this is from Berlin', location={'lon': 13.24, 'lat': 50.31}),
    NewsDoc(text='this is from Beijing', location={'lon': 116.22, 'lat': 39.55}),
    NewsDoc(text='this is from San Jose', location={'lon': -121.89, 'lat': 37.34}),
]
doc_index.index(index_docs)

# filter the eastern hemisphere
query = {
    'bool': {
        'filter': {
            'geo_bounding_box': {
                'location': {
                    'top_left': {'lon': 0, 'lat': 90},
                    'bottom_right': {'lon': 180, 'lat': 0},
                }
            }
        }
    }
}

docs = doc_index.filter(query)
```

### Range filter

You can have [range field types](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/range.html) in your document schema and set `Field(col_type='integer_range')`(or also `date_range`, etc.) to filter documents based on the range of the field. 

```python
class NewsDoc(BaseDoc):
    time_frame: dict = Field(
        col_type='date_range', format='yyyy-MM-dd'
    )  # enable range filtering


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(time_frame={'gte': '2023-01-01', 'lt': '2023-02-01'}),
    NewsDoc(time_frame={'gte': '2023-02-01', 'lt': '2023-03-01'}),
    NewsDoc(time_frame={'gte': '2023-03-01', 'lt': '2023-04-01'}),
]
doc_index.index(index_docs)

query = {
    'bool': {
        'filter': {
            'range': {
                'time_frame': {
                    'gte': '2023-02-05',
                    'lt': '2023-02-10',
                    'relation': 'contains',
                }
            }
        }
    }
}

docs = doc_index.filter(query)
```


## Text search

In addition to vector similarity search, the Document Index interface offers methods for text search:
[`text_search()`][docarray.index.abstract.BaseDocIndex.text_search],
as well as the batched version [`text_search_batched()`][docarray.index.abstract.BaseDocIndex.text_search_batched].

As in "pure" Elasticsearch, you can use text search directly on the field of type `str`:

```python
class NewsDoc(BaseDoc):
    text: str


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(id='0', text='this is a news for sport'),
    NewsDoc(id='1', text='this is a news for finance'),
    NewsDoc(id='2', text='this is another news for sport'),
]
doc_index.index(index_docs)
query = 'finance'

# search with text
docs, scores = doc_index.text_search(query, search_field='text')
```


## Hybrid search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [`build_query()`][docarray.index.abstract.BaseDocIndex.build_query]:

For example, you can build a hybrid serach query that performs range filtering, vector search and text search:

```python
class MyDoc(BaseDoc):
    tens: NdArray[10] = Field(similarity='l2_norm')
    num: int
    text: str


doc_index = ElasticDocIndex[MyDoc]()
index_docs = [
    MyDoc(id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'text {int(i/2)}')
    for i in range(10)
]
doc_index.index(index_docs)

q = (
    doc_index.build_query()
    .filter({'range': {'num': {'lte': 3}}})
    .find(index_docs[-1], search_field='tens')
    .text_search('0', search_field='text')
    .build()
)
docs, _ = doc_index.execute_query(q)
```

You can also manually build a valid ES query and directly pass it to the `execute_query()` method.


## Access documents

To access a document, you need to specify its `id`. You can also pass a list of `id`s to access multiple documents.

```python
# access a single Doc
doc_index[index_docs[1].id]

# access multiple Docs
doc_index[index_docs[2].id, index_docs[3].id]
```

## Delete documents

To delete documents, use the built-in function `del` with the `id` of the documents that you want to delete.
You can also pass a list of `id`s to delete multiple documents.

```python
# delete a single Doc
del doc_index[index_docs[1].id]

# delete multiple Docs
del doc_index[index_docs[2].id, index_docs[3].id]
```


## Configuration

### DBConfig

The following configs can be set in `DBConfig`:

| Name              | Description                                                                                                                            | Default                 |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| `hosts`           | Hostname of the Elasticsearch server                                                                                                   | `http://localhost:9200` |
| `es_config`       | Other ES [configuration options](https://www.elastic.co/guide/en/elasticsearch/client/python-api/8.6/config.html) in a Dict and pass to `Elasticsearch` client constructor, e.g. `cloud_id`, `api_key` | `None` |
| `index_name`      | Elasticsearch index name, the name of Elasticsearch index object                                       | `None`. Data will be stored in an index named after the Document type used as schema. |
| `index_settings`  | Other [index settings](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/index-modules.html#index-modules-settings) in a Dict for creating the index    | dict  |
| `index_mappings`  | Other [index mappings](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/mapping.html) in a Dict for creating the index | dict  |
| `default_column_config`  | The default configurations for every column type. | dict  |

You can pass any of the above as keyword arguments to the `__init__()` method or pass an entire configuration object.
See [here](docindex.md#configuration-options#customize-configurations) for more information.

`default_column_config` is the default configurations for every column type. Since there are many column types in Elasticsearch, you can also consider changing the column config when defining the schema.

```python
class SimpleDoc(BaseDoc):
    tensor: NdArray[128] = Field(similarity='l2_norm', m=32, num_candidates=5000)


doc_index = ElasticDocIndex[SimpleDoc](index_name='my_index_1')
```

### RuntimeConfig

The `RuntimeConfig` dataclass of `ElasticDocIndex` consists of `chunk_size`. You can change `chunk_size` for batch operations:

```python
doc_index = ElasticDocIndex[SimpleDoc](index_name='my_index_2')
doc_index.configure(ElasticDocIndex.RuntimeConfig(chunk_size=1000))
```

You can pass the above as keyword arguments to the `configure()` method or pass an entire configuration object.
See [here](docindex.md#configuration-options#customize-configurations) for more information.


### Persistence

You can hook into a database index that was persisted during a previous session by 
specifying the `index_name` and `hosts`:

```python
doc_index = ElasticDocIndex[MyDoc](
    hosts='http://localhost:9200', index_name='previously_stored'
)
doc_index.index(index_docs)

doc_index2 = ElasticDocIndex[MyDoc](
    hosts='http://localhost:9200', index_name='previously_stored'
)

print(f'number of docs in the persisted index: {doc_index2.num_docs()}')
```


## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such as `str` or `NdArray`. 
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a document 
contains a `DocList` of other documents. 

Go to the [Nested Data](nested_data.md) section to learn more.