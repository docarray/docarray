(data-management)=
## Data Management
Jina AI Cloud offers data management of DocumentArrays, either using the console or the Python API of DocArray.

### Web Console
In order to use the web console to manage your storage, you need to login at [cloud.jina.ai](https://cloud.jina.ai).
Then, head out to the [User Storage page](https://cloud.jina.ai/user/storage).

Your DocumentArrays should appear in the data section inside the storage page:
```{figure} user-storage-page.png
:width: 90%
```

You can delete, download, view or change the visibility of your DocumentArray objects using the web console.


### Python API
Data management using the Python API is offered by DocArray.
Once you've successfully {ref}`logged in<login>`, you can start using `DocumentArray` methods to manage data.

#### Push (create/update):
You can push in-memory `DocumentArray` objects using the method {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.push`:
```python
from docarray import DocumentArray

da = DocumentArray(...)
da.push('my_da', show_progress=True)
```
This will create a DocumentArray object in the cloud or update it if it already exists.

#### Pull (Read):
You can download a DocumentArray stored in the cloud using the method {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.pull`:
```python
from docarray import DocumentArray

my_da = DocumentArray('my_da')
```

#### List
It is possible to list all `DocumentArray` objects stored in the cloud using the method {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.cloud_list`: 
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
Use parameter `show_table` to show table summarizing information about DocumentArrays in the cloud.
```

#### Delete

It is also possible to delete DocumentArray objects in the cloud using the method {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.cloud_delete`:
```python
DocumentArray.cloud_delete('my_da')
```
