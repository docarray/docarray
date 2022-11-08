(interaction-cloud)=
# Interaction with Jina AI Cloud

```{important}
This feature requires the `rich` and `requests` dependencies. You can do `pip install "docarray[full]"` to install them.
```

The {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.push` and {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.pull` methods allow you to serialize a DocumentArray object to Jina AI Cloud and share it across machines.

Imagine you're working on a GPU machine via Google Colab/Jupyter. After preprocessing and embedding, you have everything you need in a DocumentArray. You can easily store it to the cloud via:

```python
from docarray import DocumentArray

da = DocumentArray(...)  # heavy lifting, processing, GPU tasks...
da.push('myda123', show_progress=True)
```

```{figure} images/da-push.png

```

Then on your local laptop, simply pull it:

```python
from docarray import DocumentArray

da = DocumentArray.pull('myda123', show_progress=True)
```

Now you can continue your work locally, analyzing `da` or visualizing it. Your friends & colleagues who know the token `myda123` can also pull that DocumentArray. It's useful when you want to quickly share the results with your colleagues & friends.

The maximum size of an upload is 4GB under the `protocol='protobuf'` and `compress='gzip'` settings. The lifetime of an upload is one week after its creation.

To avoid unnecessary downloads when the upstream DocumentArray is unchanged, you can add `DocumentArray.pull(..., local_cache=True)`.

```{seealso}
DocArray allows pushing, pulling, and managing your DocumentArrays in Jina AI Cloud.
Read more about how to manage your data in Jina AI Cloud, using either the console or the DocArray Python API, in the
{ref}`Data Management section <data-management>`.
```
