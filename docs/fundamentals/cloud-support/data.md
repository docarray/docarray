(data-management)=
# Data Management
Jina AI Cloud offers data management of DocumentArrays, using either the console or the DocArray Python API.

## Web Console
In order to use the web console to manage your storage, you need to log in at [cloud.jina.ai](https://cloud.jina.ai).
Then, head to the [User Storage page](https://cloud.jina.ai/user/storage).

Your DocumentArrays should appear in the data section inside the storage page:
```{figure} user-storage-page.png
:width: 90%
```

You can delete, download, view, or change the visibility of your DocumentArray objects using the web console.


## Python API
DocArray offers a Python API for data management.
Once you've successfully {ref}`logged in<login>`, you can start using `DocumentArray` methods to manage data.

### Push (create/update):
You can push in-memory `DocumentArray` objects using the {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.push` method:
```python
from docarray import DocumentArray

da = DocumentArray(...)
da.push('my_da', show_progress=True)
```
This will create a DocumentArray object in the cloud or update it if it already exists.

### Pull (Read):
You can download a DocumentArray stored in the cloud using the {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.pull` method:
```python
from docarray import DocumentArray

my_da = DocumentArray('my_da')
```

### List
You can list all `DocumentArray` objects stored in the cloud using the {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.cloud_list` method: 
```python
DocumentArray.cloud_list(show_table=True)
```

```text
                      You have 1 DocumentArray on the cloud                       
                                                                                  
  Name     Length   Access          Created at                 Updated at         
 ──────────────────────────────────────────────────────────────────────────────── 
  my_da    10       public   2022-09-15T07:14:54.256Z   2022-09-15T07:14:54.256Z  
                                                                                  
['my_da']
```

```{tip}
Use the `show_table` parameter to show summary information about DocumentArrays in the cloud.
```

### Delete

You can delete DocumentArray objects in the cloud using the method {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.cloud_delete`:
```python
DocumentArray.cloud_delete('my_da')
```
