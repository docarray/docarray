# Weaviate Document Index

!!! note "Install dependencies"
    To use [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[weaviate]"
    ```

This is the user guide for the [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex],
focusing on special features and configurations of Weaviate.


## Basic usage
This snippet demonstrates the basic usage of [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex]. It defines a document schema with a title and an embedding, 
creates ten dummy documents with random embeddings, initializes an instance of [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex] to index these documents, 
and performs a vector similarity search to retrieve ten most similar documents to a given query vector.

!!! note "Single Search Field Requirement"
    In order to utilize vector search, it's necessary to define 'is_embedding' for one field only. 
    This is due to Weaviate's configuration, which permits a single vector for each data object.

```python
from docarray import BaseDoc, DocList
from docarray.index import WeaviateDocumentIndex
from docarray.typing import NdArray
from pydantic import Field
import numpy as np

# Define the document schema.
class MyDoc(BaseDoc):
    title: str 
    embedding: NdArray[128] = Field(is_embedding=True)

# Create dummy documents.
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))

# Initialize a new WeaviateDocumentIndex instance and add the documents to the index.
doc_index = WeaviateDocumentIndex[MyDoc]()
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, limit=10)
```


## Initialize


### Start Weaviate service

To use [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex], DocArray needs to hook into a running Weaviate service.
There are multiple ways to start a Weaviate instance, depending on your use case.

**Options - Overview**

| Instance type | General use case | Configurability | Notes | 
| ----- | ----- | ----- | ----- | 
| **Weaviate Cloud Services (WCS)** | Development and production | Limited | **Recommended for most users** |
| **Embedded Weaviate** | Experimentation | Limited | Experimental (as of Apr 2023) |
| **Docker-Compose** | Development | Yes | **Recommended for development + customizability** |
| **Kubernetes** | Production | Yes |  |

### Instantiation instructions

**WCS (managed instance)**

Go to the [WCS console](https://console.weaviate.cloud) and create an instance using the visual interface, following [this guide](https://weaviate.io/developers/wcs/guides/create-instance). 

Weaviate instances on WCS come pre-configured, so no further configuration is required.

**Docker-Compose (self-managed)**

Get a configuration file (`docker-compose.yaml`). You can build it using [this interface](https://weaviate.io/developers/weaviate/installation/docker-compose), or download it directly with:

```bash
curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v<WEAVIATE_VERSION>"
```

Where `v<WEAVIATE_VERSION>` is the actual version, such as `v1.18.3`.

```bash
curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.18.3"
```

**Start up Weaviate with Docker-Compose**

Then you can start up Weaviate by running from a shell:

```shell
docker-compose up -d
```

**Shut down Weaviate**

Then you can shut down Weaviate by running from a shell:

```shell
docker-compose down
```

**Notes**

Unless data persistence or backups are set up, shutting down the Docker instance will remove all its data. 

See documentation on [Persistent volume](https://weaviate.io/developers/weaviate/installation/docker-compose#persistent-volume) and [Backups](https://weaviate.io/developers/weaviate/configuration/backups) to prevent this if persistence is desired.

```bash
docker-compose up -d
```

**Embedded Weaviate (from the application)**

With Embedded Weaviate, Weaviate database server can be launched from the client, using:

```python
from docarray.index.backends.weaviate import EmbeddedOptions

embedded_options = EmbeddedOptions()
```

### Authentication

Weaviate offers [multiple authentication options](https://weaviate.io/developers/weaviate/configuration/authentication), as well as [authorization options](https://weaviate.io/developers/weaviate/configuration/authorization). 

With DocArray, you can use any of:

- Anonymous access (public instance),
- OIDC with username & password, and
- API-key based authentication.

To access a Weaviate instance. In general, **Weaviate recommends using API-key based authentication** for balance between security and ease of use. You can create, for example, read-only keys to distribute to certain users, while providing read/write keys to administrators.

See below for examples of connection to Weaviate for each scenario.

### Connect to Weaviate

```python
from docarray.index.backends.weaviate import WeaviateDocumentIndex
```

### Public instance

If using Embedded Weaviate:

```python
from docarray.index.backends.weaviate import EmbeddedOptions

dbconfig = WeaviateDocumentIndex.DBConfig(embedded_options=EmbeddedOptions())
```

For all other options:

```python
dbconfig = WeaviateDocumentIndex.DBConfig(
    host="http://localhost:8080"
)  # Replace with your endpoint)
```

**OIDC with username + password**

To authenticate against a Weaviate instance with OIDC username & password:

```python
dbconfig = WeaviateDocumentIndex.DBConfig(
    username="username",  # Replace with your username
    password="password",  # Replace with your password
    host="http://localhost:8080",  # Replace with your endpoint
)
```

```python
# dbconfig = WeaviateDocumentIndex.DBConfig(
#     username="username",  # Replace with your username
#     password="password",  # Replace with your password
#     host="http://localhost:8080",  # Replace with your endpoint
# )
```

**API key-based authentication**

To authenticate against a Weaviate instance an API key:

```python
dbconfig = WeaviateDocumentIndex.DBConfig(
    auth_api_key="apikey",  # Replace with your own API key
    host="http://localhost:8080",  # Replace with your endpoint
)
```

### Create an instance
Let's connect to a local Weaviate service and instantiate a `WeaviateDocumentIndex` instance:
```python
dbconfig = WeaviateDocumentIndex.DBConfig(
    host="http://localhost:8080"
)
doc_index = WeaviateDocumentIndex[MyDoc](db_config=dbconfig)
```

### Schema definition
In this code snippet, `WeaviateDocumentIndex` takes a schema of the form of `MyDoc`.
The Document Index then _creates a column for each field in `MyDoc`_.

The column types in the backend database are determined by the type hints of the document's fields.
Optionally, you can [customize the database types for every field](#configuration).

Most vector databases need to know the dimensionality of the vectors that will be stored.
Here, that is automatically inferred from the type hint of the `embedding` field: `NdArray[128]` means that
the database will store vectors with 128 dimensions.

!!! note "PyTorch and TensorFlow support"
    Instead of using `NdArray` you can use `TorchTensor` or `TensorFlowTensor` and the Document Index will handle that
    for you. This is supported for all Document Index backends. No need to convert your tensors to NumPy arrays manually!

### Using a predefined document as schema

DocArray offers a number of predefined documents, like [ImageDoc][docarray.documents.ImageDoc] and [TextDoc][docarray.documents.TextDoc].
If you try to use these directly as a schema for a Document Index, you will get unexpected behavior:
Depending on the backend, an exception will be raised, or no vector index for ANN lookup will be built.

The reason for this is that predefined documents don't hold information about the dimensionality of their `.embedding`
field. But this is crucial information for any vector database to work properly!

You can work around this problem by subclassing the predefined document and adding the dimensionality information:

=== "Using type hint"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import NdArray
    from docarray.index import WeaviateDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: NdArray[128] = Field(is_embedding=True)


    doc_index = WeaviateDocumentIndex[MyDoc]()
    ```

=== "Using Field()"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyTensor
    from docarray.index import WeaviateDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: AnyTensor = Field(dim=128, is_embedding=True)


    doc_index = WeaviateDocumentIndex[MyDoc]()
    ```

Once you have defined the schema of your Document Index in this way, the data that you index can be either the predefined Document type or your custom Document type.

The [next section](#index) goes into more detail about data indexing, but note that if you have some `TextDoc`s, `ImageDoc`s etc. that you want to index, you _don't_ need to cast them to `MyDoc`:

```python
from docarray import DocList

# data of type TextDoc
data = DocList[TextDoc](
    [
        TextDoc(text='hello world', embedding=np.random.rand(128)),
        TextDoc(text='hello world', embedding=np.random.rand(128)),
        TextDoc(text='hello world', embedding=np.random.rand(128)),
    ]
)

# you can index this into Document Index of type MyDoc
doc_index.index(data)
```


## Index

Putting it together, we can add data below using Weaviate as the Document Index:

```python
import numpy as np
from pydantic import Field
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from docarray.index.backends.weaviate import WeaviateDocumentIndex


# Define a document schema
class Document(BaseDoc):
    text: str
    embedding: NdArray[2] = Field(
        dims=2, is_embedding=True
    )  # Embedding column -> vector representation of the document
    file: NdArray[100] = Field(dims=100)


# Make a list of 3 docs to index
docs = DocList[Document](
    [
        Document(
            text="Hello world",
            embedding=np.array([1, 2]),
            file=np.random.rand(100),
            id="1",
        ),
        Document(
            text="Hello world, how are you?",
            embedding=np.array([3, 4]),
            file=np.random.rand(100),
            id="2",
        ),
        Document(
            text="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut",
            embedding=np.array([5, 6]),
            file=np.random.rand(100),
            id="3",
        ),
    ]
)

batch_config = {
    "batch_size": 20,
    "dynamic": False,
    "timeout_retries": 3,
    "num_workers": 1,
}

runtimeconfig = WeaviateDocumentIndex.RuntimeConfig(batch_config=batch_config)

store = WeaviateDocumentIndex[Document]()
store.configure(runtimeconfig)  # Batch settings being passed on
store.index(docs)
```

### Notes

- To use vector search, you need to specify `is_embedding` for exactly one field.
    - This is because Weaviate is configured to allow one vector per data object.
    - If you would like to see Weaviate support multiple vectors per object, [upvote the issue](https://github.com/weaviate/weaviate/issues/2465) which will help to prioritize it.
- For a field to be considered as an embedding, its type needs to be of subclass `np.ndarray` or `AbstractTensor` and `is_embedding` needs to be set to `True`. 
    - If `is_embedding` is set to `False` or not provided, the field will be treated as a `number[]`, and as a result, it will not be added to Weaviate's vector index.
- It is possible to create a schema without specifying `is_embedding` for any field. 
    - This will however mean that the document will not be vectorized and cannot be searched using vector search. 

As you can see, `DocList[Document]` and `WeaviateDocumentIndex[Document]` both have `Document` as a parameter.
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

You can perform a similarity search and find relevant documents by passing `MyDoc` or a raw vector to 
the [`find()`][docarray.index.abstract.BaseDocIndex.find] method:

=== "Search by Document"

    ```python
    # create a query document
    query = Document(
        text="Hello world",
        embedding=np.array([1, 2]),
        file=np.random.rand(100),
    )
    
    # find similar documents
    matches, scores = doc_index.find(query, limit=5)
    
    print(f"{matches=}")
    print(f"{matches.text=}")
    print(f"{scores=}")
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(2)

    # find similar documents
    matches, scores = store.find(query, limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

In this example you only have one field (`embedding`) that is a vector, so you can trivially choose that one.
In general, you could have multiple fields of type `NdArray` or `TorchTensor` or `TensorFlowTensor`, and you can choose
which one to use for the search.

The [`find()`][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on the subindex level, you can use the [`find_subindex()`][docarray.index.abstract.BaseDocIndex.find_subindex] method, which returns a named tuple containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#configuration).

### Batched search

You can also search for multiple documents at once, in a batch, using the [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method.

=== "Search by documents"

    ```python
    # create some query documents
    queries = DocList[MyDoc](
        Document(
            text=f"Hello world {i}",
            embedding=np.array([i, i + 1]),
            file=np.random.rand(100),
        )
        for i in range(3)
    )
    
    # find similar documents
    matches, scores = doc_index.find_batched(queries, limit=5)
    
    print(f"{matches=}")
    print(f"{matches[0].text=}")
    print(f"{scores=}")
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 2)

    # find similar documents
    matches, scores = doc_index.find_batched(query, limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.


## Filter

To perform filtering, follow the below syntax. 

This will perform a filtering on the field `text`:
```python
docs = store.filter({"path": ["text"], "operator": "Equal", "valueText": "Hello world"})
```

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding  filter query. 
The query should follow the [query language of the Weaviate](https://weaviate.io/developers/weaviate/search/filters).

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from pydantic import Field
import numpy as np


class Book(BaseDoc):
    price: int
    embedding: NdArray[10] = Field(is_embedding=True)


books = DocList[Book]([Book(price=i * 10, embedding=np.random.rand(10)) for i in range(10)])
book_index = WeaviateDocumentIndex[Book](index_name='tmp_index')
book_index.index(books)

# filter for books that are cheaper than 29 dollars
query = {"path": ["price"], "operator": "LessThan", "valueInt": 29}
cheap_books = book_index.filter(filter_query=query)

assert len(cheap_books) == 3
for doc in cheap_books:
    doc.summary()
```


## Text search

In addition to vector similarity search, the Document Index interface offers methods for text search:
[`text_search()`][docarray.index.abstract.BaseDocIndex.text_search],
as well as the batched version [`text_search_batched()`][docarray.index.abstract.BaseDocIndex.text_search_batched].

You can use text search directly on the field of type `str`.

The following line will perform a text search for the word "hello" in the field "text" and return the first two results:

```python
docs = store.text_search("world", search_field="text", limit=2)
```


## Hybrid search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [`build_query()`][docarray.index.abstract.BaseDocIndex.build_query].

To perform a hybrid search, follow the below syntax. 

This will perform a hybrid search for the word "hello" and the vector [1, 2] and return the first two results:

**Note**: Hybrid search searches through the object vector and all fields. Accordingly, the `search_field` keyword will have no effect. 

```python
q = store.build_query().text_search("world").find([1, 2]).limit(2).build()

docs = store.execute_query(q)
```

### GraphQL query

You can also perform a raw GraphQL query using any syntax as you might natively in Weaviate. This allows you to run any of the full range of queries that you might wish to.

The below will perform a GraphQL query to obtain the count of `Document` objects. 

```python
graphql_query = """
{
  Aggregate {
    Document {
      meta {
        count
      }
    }
  }
}
"""

store.execute_query(graphql_query)
```

Note that running a raw GraphQL query will return Weaviate-type responses, rather than a DocArray object type.

You can find the documentation for [Weaviate's GraphQL API here](https://weaviate.io/developers/weaviate/api/graphql).

## Access documents

To retrieve a document from a Document Index you don't necessarily need to perform a fancy search.

You can also access data by the `id` that was assigned to each document:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), title=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the documents by id
doc = doc_index[ids[0]]  # get by single id
docs = doc_index[ids]  # get by list of ids
```


## Delete documents

In the same way you can access documents by `id`, you can also delete them:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), title=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the documents by id
del doc_index[ids[0]]  # del by single id
del doc_index[ids[1:]]  # del by list of ids
```

## Configuration

### Overview

**WCS instances come pre-configured**, and as such additional settings are not configurable outside of those chosen at creation, such as whether to enable authentication.

For other cases, such as **Docker-Compose deployment**, its settings can be modified through the configuration file, such as the `docker-compose.yaml` file. 

Some of the more commonly used settings include:

- [Persistent volume](https://weaviate.io/developers/weaviate/installation/docker-compose#persistent-volume): Set up data persistence so that data from inside the Docker container is not lost on shutdown
- [Enabling a multi-node setup](https://weaviate.io/developers/weaviate/installation/docker-compose#multi-node-setup)
- [Backups](https://weaviate.io/developers/weaviate/configuration/backups)
- [Authentication (server-side)](https://weaviate.io/developers/weaviate/configuration/authentication)
- [Modules enabled](https://weaviate.io/developers/weaviate/configuration/modules#enable-modules)

And a list of environment variables is [available on this page](https://weaviate.io/developers/weaviate/config-refs/env-vars).

### DocArray instantiation configuration options

Additionally, you can specify the below settings when you instantiate a configuration object in DocArray.

| name | type | explanation | default                                                                | example |
| ---- | ---- | ----------- |------------------------------------------------------------------------| ------- |
| **Category: General** |
| host | str | Weaviate instance url | http://localhost:8080                                                  |
| **Category: Authentication** |
| username | str | Username known to the specified authentication provider (e.g. WCS) | `None`                                                                   | `jp@weaviate.io` |
| password | str | Corresponding password | `None`                                                                   | `p@ssw0rd` |
| auth_api_key | str | API key known to the Weaviate instance | `None`                                                                   | `mys3cretk3y` | 
| **Category: Data schema** |
| index_name | str | Class name to use to store the document| The document class name, e.g. `MyDoc` for `WeaviateDocumentIndex[MyDoc]` | `Document` |
| **Category: Embedded Weaviate** |
| embedded_options| EmbeddedOptions | Options for embedded weaviate | `None`                                                                   |

The type `EmbeddedOptions` can be specified as described [here](https://weaviate.io/developers/weaviate/installation/embedded#embedded-options)

### Runtime configuration

Weaviate strongly recommends using batches to perform bulk operations such as importing data, as it will significantly impact performance. You can specify a batch configuration as in the below example, and pass it on as runtime configuration.

```python
batch_config = {
    "batch_size": 20,
    "dynamic": False,
    "timeout_retries": 3,
    "num_workers": 1,
}

runtimeconfig = WeaviateDocumentIndex.RuntimeConfig(batch_config=batch_config)

dbconfig = WeaviateDocumentIndex.DBConfig(
    host="http://localhost:8080"
)  # Replace with your endpoint and/or auth settings
store = WeaviateDocumentIndex[Document](db_config=dbconfig)
store.configure(runtimeconfig)  # Batch settings being passed on
```

| name | type | explanation | default |
| ---- | ---- | ----------- | ------- |
| batch_config | Dict[str, Any] | dictionary to configure the weaviate client's batching logic | see below |

Read more: 

- Weaviate [docs on batching with the Python client](https://weaviate.io/developers/weaviate/client-libraries/python#batching)


### Available column types

Python data types are mapped to Weaviate type according to the below conventions.

| Python type | Weaviate type |
| ----------- | ------------- |
| docarray.typing.ID | string |
| str | text |
| int | int |
| float | number |
| bool | boolean |
| np.ndarray | number[] |
| AbstractTensor | number[] |
| bytes | blob |

You can override this default mapping by passing a `col_type` to the `Field` of a schema. 

For example to map `str` to `string` you can:

```python
class StringDoc(BaseDoc):
    text: str = Field(col_type="string")
```

A list of available Weaviate data types [is here](https://weaviate.io/developers/weaviate/config-refs/datatypes).


## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such as `str` or `NdArray`. 
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a document 
contains a `DocList` of other documents. 

Go to the [Nested Data](nested_data.md) section to learn more.