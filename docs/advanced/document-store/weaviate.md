# Weaviate

One can use [Weaviate](https://www.semi.technology) as the document store for DocumentArray. It is useful when one wants to leverage Weaviate for storage and vector search.


## Start Weaviate Service

To use Weaviate as storage backend, one is required to have the Weaviate service started.

To start the Weaviate Docker Service, simply follow the instructions detailed [here](https://www.semi.technology/developers/weaviate/current/getting-started/installation.html#weaviate-without-any-modules).

## Usage

Assuming service is started using the default configuration (i.e. server address is `http://localhost:8080`), one can instantiate a `DocumentArray` with Weaviate storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='weaviate', config={'n_dim': 10})
```

```{admonition} Config is Required
:class: tip
Unlike SQLite storage, `config` is a required parameter to instantiate `DocumentArray` with Weaviate as storage.
In `config`, user is required to provide an `int` value for `n_dim` that represents the number of dimension of embeddings to be stored.
This allows for Weaviate's vector search capability.
```

To access `DocumentArray` formerly persisted in Weaviate, one can specify the name of the persisted Weaviate object representing the `DocumentArray`,
along with the address or the client connecting to the server where data is persisted (`name` is required in this case but `client` is optional.
If `client` is not provided, then it will connect to the Weaviate service bound to `http://localhost:8080`).

Note, that the `name` parameter in `config` needs to be capitalized.

```python
from docarray import DocumentArray

da = DocumentArray(storage='weaviate', config={'name': 'Persisted', 'client': 'http://localhost:1234', 'n_dim': 10})

da.summary()
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name               | Description                                                                                             | Default                     |
|--------------------|---------------------------------------------------------------------------------------------------------|-----------------------------|
| `n_dim`            | Number of dimensions of embeddings to be stored and retrieved                                           | N/A, this is required field |
| `client`           | Weaviate client; this can be a string uri representing the server address or a `weaviate.Client` object | `'http://localhost:8080'`   |
| `name`             | Weaviate class name; the class name of Weaviate object to presesent this DocumentArray                  | None                        |
| `serialize_config` | [Serialization config of each Document](../../fundamentals/document/serialization.md)                   | None                        |
