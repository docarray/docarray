# Store on S3
When you want to use your [`DocList`][docarray.DocList] in another place, you can use the 
[`.push`][docarray.array.doc_list.pushpull.PushPullMixin.push] method to push the `DocList` to S3 and later use the
[`.pull`][docarray.array.doc_list.pushpull.PushPullMixin.pull] function to pull its content back. 

!!! note
    To store on S3, you need to install the extra dependency with the following line
    ```cmd
    pip install "docarray[aws]"
    ```

## Push & pull
To use the store [`DocList`][docarray.DocList] on S3, you need to pass an S3 path to the function starting with `'s3://'`.

In the following demo, we use `MinIO` as a local S3 service. You could use the following docker compose file to start the service in a Docker container.

```yaml
version: "3"
services:
  minio:
    container_name: minio
    image: "minio/minio:RELEASE.2023-03-13T19-46-17Z"
    ports:
      - "9005:9000"
    command: server /data
```
Save the above file as `docker-compose.yml` and run the following line in the same folder as the file.
```cmd
docker compose up
```

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


if __name__ == '__main__':
    import boto3
    from botocore.client import Config

    BUCKET = 'tmp_bucket'
    my_session = boto3.session.Session()
    s3 = my_session.resource(
        service_name='s3',
        region_name="us-east-1",
        use_ssl=False,
        endpoint_url="http://localhost:9005",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
    )
    # make a bucket
    s3.create_bucket(Bucket=BUCKET)

    store_docs = [SimpleDoc(text=f'doc {i}') for i in range(8)]
    docs = DocList[SimpleDoc]()
    docs.extend([SimpleDoc(text=f'doc {i}') for i in range(8)])

    # .push() and .pull() use the default boto3 client
    boto3.Session.client.__defaults__ = (
        "us-east-1",
        None,
        False,
        None,
        "http://localhost:9005",
        "minioadmin",
        "minioadmin",
        None,
        Config(signature_version="s3v4"),
    )
    docs.push(f's3://{BUCKET}/simple_docs')
    docs_pull = DocList[SimpleDoc].pull(f's3://{BUCKET}/simple_docs')
```

Under the bucket `tmp_bucket`, there is a file with the name of `simple_docs.docs` being created to store the `DocList`.

!!! note
    When using `.push()` and `.pull()`, `DocList` calls the default boto3 client. Be sure your default session is correctly set up.


## Push & pull with streaming
When you have a large amount of documents to push and pull, you could use the streaming function. 
[`.push_stream()`][docarray.array.doc_list.pushpull.PushPullMixin.push_stream] and 
[`.pull_stream()`][docarray.array.doc_list.pushpull.PushPullMixin.pull_stream] can help you to stream the 
[`DocList`][docarray.DocList] in order to save the memory usage. You set multiple [`DocList`][docarray.DocList] to pull from the same source as well. The usage is the same as using streaming with local files. Please refer to [Push & Pull with streaming with local files](store_file.md#push-pull-with-streaming).


## Delete
To delete the store, you need to use the static method [`.delete()`][docarray.store.s3.S3DocStore.delete] of [`S3DocStore`][docarray.store.s3.S3DocStore] class.

```python hl_lines="44-47"
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


if __name__ == '__main__':
    import boto3
    from botocore.client import Config

    BUCKET = 'tmp_bucket'
    my_session = boto3.session.Session()
    s3 = my_session.resource(
        service_name='s3',
        region_name="us-east-1",
        use_ssl=False,
        endpoint_url="http://localhost:9005",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
    )
    # make a bucket
    s3.create_bucket(Bucket=BUCKET)

    store_docs = [SimpleDoc(text=f'doc {i}') for i in range(8)]
    docs = DocList[SimpleDoc]()
    docs.extend([SimpleDoc(text=f'doc {i}') for i in range(8)])

    # .push() and .pull() use the default boto3 client
    boto3.Session.client.__defaults__ = (
        "us-east-1",
        None,
        False,
        None,
        "http://localhost:9005",
        "minioadmin",
        "minioadmin",
        None,
        Config(signature_version="s3v4"),
    )
    docs.push(f's3://{BUCKET}/simple_docs')

    # delete bucket
    from docarray.store import S3DocStore

    success = S3DocStore.delete('{BUCKET}/simple_docs')
```
