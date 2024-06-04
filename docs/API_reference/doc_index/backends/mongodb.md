# MongoDBAtlasDocumentIndex

::: docarray.index.backends.mongodb_atlas.MongoDBAtlasDocumentIndex

# Setting up MongoDB Atlas as the Document Index

MongoDB Atlas is a multi-cloud database service made by the same people that build MongoDB. 
Atlas simplifies deploying and managing your databases while offering the versatility you need 
to build resilient and performant global applications on the cloud providers of your choice.

You can perform semantic search on data in your Atlas cluster running MongoDB v6.0.11 
or later using Atlas Vector Search. You can store vector embeddings for any kind of data along 
with other data in your collection on the Atlas cluster.

In the section, we set up a cluster, a database, test it, and finally create an Atlas Vector Search Index.

### Deploy a Cluster

Follow the [Getting-Started](https://www.mongodb.com/basics/mongodb-atlas-tutorial) documentation 
to create an account, deploy an Atlas cluster, and connect to a database.


### Retrieve the URI used by Python to connect to the Cluster

When you deploy, this will be stored as the environment variable: `MONGODB_URI`  
It will look something like the following. The username and password, if not provided,
can be configured in *Database Access* under Security in the left panel. 

```
export MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.foo.mongodb.net/?retryWrites=true&w=majority"
```

There are a number of ways to navigate the Atlas UI. Keep your eye out for "Connect" and "Driver".

On the left panel, navigate and click 'Database' under DEPLOYMENT. 
Click the Connect button that appears, then Drivers. Select Python.
(Have no concern for the version. This is the PyMongo, not Python, version.)
Once you have got the Connect Window open, you will see an instruction to `pip install pymongo`.
You will also see a **connection string**. 
This is the `uri` that a `pymongo.MongoClient` uses to connect to the Database.


### Test the connection

Atlas provides a simple check. Once you have your `uri` and `pymongo` installed, 
try the following in a python console.

```python
from pymongo.mongo_client import MongoClient
client = MongoClient(uri)  # Create a new client and connect to the server
try:
    client.admin.command('ping')  # Send a ping to confirm a successful connection
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
```

**Troubleshooting**
* You can edit a Database's users and passwords on the 'Database Access' page, under Security.
* Remember to add your IP address. (Try `curl -4 ifconfig.co`)

### Create a Database and Collection

As mentioned, Vector Databases provide two functions. In addition to being the data store,
they provide very efficient search based on natural language queries.
With Vector Search, one will index and query data with a powerful vector search algorithm
using "Hierarchical Navigable Small World (HNSW) graphs to find vector similarity.

The indexing runs beside the data as a separate service asynchronously.
The Search index monitors changes to the Collection that it applies to.
Subsequently, one need not upload the data first. 
We will create an empty collection now, which will simplify setup in the example notebook.

Back in the UI, navigate to the Database Deployments page by clicking Database on the left panel.
Click the "Browse Collections" and then "+ Create Database" buttons. 
This will open a window where you choose Database and Collection names. (No additional preferences.)
Remember these values as they will be as the environment variables, 
`MONGODB_DATABASE`.

### MongoDBAtlasDocumentIndex

To connect to the MongoDB Cluster and Database, define the following environment variables.
You can confirm that the required ones have been set like this:  `assert "MONGODB_URI" in os.environ`

**IMPORTANT** It is crucial that the choices are consistent between setup in Atlas and Python environment(s).

| Name                  | Description                 | Example                                                      |
|-----------------------|-----------------------------|--------------------------------------------------------------|
| `MONGODB_URI`         | Connection String           | mongodb+srv://`<user>`:`<password>`@cluster0.bar.mongodb.net |
| `MONGODB_DATABASE`    | Database name               | docarray_test_db                                             |


```python

from docarray.index.backends.mongodb_atlas import MongoDBAtlasDocumentIndex
import os

index = MongoDBAtlasDocumentIndex(
    mongo_connection_uri=os.environ["MONGODB_URI"],
    database_name=os.environ["MONGODB_DATABASE"])
```


### Create an Atlas Vector Search Index

The final step to configure a MongoDBAtlasDocumentIndex is to create a Vector Search Indexes.
The procedure is described [here](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure).

Under Services on the left panel, choose Atlas Search > Create Search Index > 
Atlas Vector Search JSON Editor. An index definition looks like the following.


```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```


### Running MongoDB Atlas Integration Tests

Setup is described in detail here `tests/index/mongo_atlas/README.md`.
There are actually a number of different collections and indexes to be created within your cluster's database.

```bash
MONGODB_URI=<uri> MONGODB_DATABASE=<db_name> py.test tests/index/mongo_atlas/
```
