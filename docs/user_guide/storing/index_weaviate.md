# Weaviate Document Index

!!! note "Install dependencies"
    To use [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[weaviate]"
    ```

This is the user guide for the [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex],
focusing on special features and configurations of Weaviate.


## Basic Usage
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

## Index

Putting it together, we can add data below using Weaviate as the Document Index:

```python
import numpy as np
from pydantic import Field
from docarray import BaseDoc
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
docs = [
    Document(
        text="Hello world", embedding=np.array([1, 2]), file=np.random.rand(100), id="1"
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


## Vector Search

To perform a vector similarity search, follow the below syntax. 

This will perform a vector similarity search for the vector [1, 2] and return the first two results:

```python
docs = store.find([1, 2], limit=2)
```

## Filter

To perform filtering, follow the below syntax. 

This will perform a filtering on the field `text`:
```python
docs = store.filter({"path": ["text"], "operator": "Equal", "valueText": "Hello world"})
```


## Text search

To perform a text search, follow the below syntax. 

This will perform a text search for the word "hello" in the field "text" and return the first two results:

```python
docs = store.text_search("world", search_field="text", limit=2)
```


## Hybrid search

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

## Access Documents

To retrieve a document from a Document Index, you don't necessarily need to perform a fancy search.

You can also access data by the `id` that was assigned to each document:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), title=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the Documents by id
doc = doc_index[ids[0]]  # get by single id
docs = doc_index[ids]  # get by list of ids
```


## Delete Documents

In the same way you can access Documents by id, you can also delete them:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), title=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the Documents by id
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
| username | str | Username known to the specified authentication provider (e.g. WCS) | None                                                                   | `jp@weaviate.io` |
| password | str | Corresponding password | None                                                                   | `p@ssw0rd` |
| auth_api_key | str | API key known to the Weaviate instance | None                                                                   | `mys3cretk3y` | 
| **Category: Data schema** |
| index_name | str | Class name to use to store the document| The document class name, e.g. `MyDoc` for `WeaviateDocumentIndex[MyDoc]` | `Document` |
| **Category: Embedded Weaviate** |
| embedded_options| EmbeddedOptions | Options for embedded weaviate | None                                                                   |

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


## Nested data

The examples above all operate on a simple schema: All fields in `MyDoc` have "basic" types, such as `str` or `NdArray`.

**Index nested data:**

It is, however, also possible to represent nested Documents and store them in a Document Index.

In the following example you can see a complex schema that contains nested Documents.
The `YouTubeVideoDoc` contains a `VideoDoc` and an `ImageDoc`, alongside some "basic" fields:

```python
from docarray.typing import ImageUrl, VideoUrl, AnyTensor


# define a nested schema
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor = Field(space='cosine', dim=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    tensor: AnyTensor = Field(space='cosine', dim=128, is_embedding=True)


class YouTubeVideoDoc(BaseDoc):
    title: str
    description: str
    thumbnail: ImageDoc
    video: VideoDoc
    tensor: AnyTensor = Field(space='cosine', dim=256)


# create a Document Index
doc_index = WeaviateDocumentIndex[YouTubeVideoDoc]()

# create some data
index_docs = [
    YouTubeVideoDoc(
        title=f'video {i+1}',
        description=f'this is video from author {10*i}',
        thumbnail=ImageDoc(url=f'http://example.ai/images/{i}', tensor=np.ones(64)),
        video=VideoDoc(url=f'http://example.ai/videos/{i}', tensor=np.ones(128)),
        tensor=np.ones(256),
    )
    for i in range(8)
]

# index the Documents
doc_index.index(index_docs)
```

**Search nested data:**

You can perform search on any nesting level by using the dunder operator to specify the field defined in the nested data.

In the following example, you can see how to perform vector search on the `tensor` field of the `video` field:

```python
# create a query Document
query_doc = YouTubeVideoDoc(
    title=f'video query',
    description=f'this is a query video',
    thumbnail=ImageDoc(url=f'http://example.ai/images/1024', tensor=np.ones(64)),
    video=VideoDoc(url=f'http://example.ai/videos/1024', tensor=np.ones(128)),
    tensor=np.ones(256),
)

# find by the `video` tensor; nested level
docs, scores = doc_index.find(query_doc, limit=3)
```

### Nested data with subindex

Documents can be nested by containing a `DocList` of other documents, which is a slightly more complicated scenario than the one [above](#nested-data).

If a Document contains a DocList, it can still be stored in a Document Index.
In this case, the DocList will be represented as a new index (or table, collection, etc., depending on the database backend), that is linked with the parent index (table, collection, ...).

This still lets index and search through all of your data, but if you want to avoid the creation of additional indexes you could try to refactor your document schemas without the use of DocList.


**Index**

In the following example you can see a complex schema that contains nested Documents with subindex.
The `MyDoc` contains a `DocList` of `VideoDoc`, which contains a `DocList` of `ImageDoc`, alongside some "basic" fields:

```python
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor_image: AnyTensor = Field(space='cosine', dim=64, is_embedding=True)


class VideoDoc(BaseDoc):
    url: VideoUrl
    images: DocList[ImageDoc]
    tensor_video: AnyTensor = Field(space='cosine', dim=128, is_embedding=True)


class MyDoc(BaseDoc):
    docs: DocList[VideoDoc]
    tensor: AnyTensor = Field(space='cosine', dim=256, is_embedding=True)


# create a Document Index
doc_index = WeaviateDocumentIndex[MyDoc]()

# create some data
index_docs = [
    MyDoc(
        docs=DocList[VideoDoc](
            [
                VideoDoc(
                    url=f'http://example.ai/videos/{i}-{j}',
                    images=DocList[ImageDoc](
                        [
                            ImageDoc(
                                url=f'http://example.ai/images/{i}-{j}-{k}',
                                tensor_image=np.ones(64),
                            )
                            for k in range(10)
                        ]
                    ),
                    tensor_video=np.ones(128),
                )
                for j in range(10)
            ]
        ),
        tensor=np.ones(256),
    )
    for i in range(10)
]

# index the Documents
doc_index.index(index_docs)
```

**Search**

You can perform search on any subindex level by using `find_subindex()` method and the dunder operator `'root__subindex'` to specify the index to search on.

```python
# find by the `VideoDoc` tensor
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(128), subindex='docs', limit=3
)

# find by the `ImageDoc` tensor
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(64), subindex='docs__images', limit=3
)
```
