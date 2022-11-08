(first-steps)=
# First steps

## Creating Documents

You can create a Document by creating a new instance of the `Document` class, and optionally pass arguments to the constructor.

```python
from docarray import Document
import numpy

d0 = Document()
d1 = Document(text='hello')
d2 = Document(blob=b'\f1')
d3 = Document(tensor=numpy.array([1, 2, 3]))
d4 = Document(
    uri='https://docarray.jina.ai',
    mime_type='text/plain',
    granularity=1,
    adjacency=3,
    tags={'foo': 'bar'},
)
```

```text
<Document ('id',) at a14effee6d3e11ec8bde1e008a366d49>
<Document ('id', 'mime_type', 'text') at a14effee6d3e11ec8bde1e008a366d49>
<Document ('id', 'blob') at a14f00986d3e11ec8bde1e008a366d49> 
<Document ('id', 'tensor') at a14f01a66d3e11ec8bde1e008a366d49> 
<Document ('id', 'granularity', 'adjacency', 'mime_type', 'uri', 'tags') at a14f023c6d3e11ec8bde1e008a366d49>
```

Every Document has a unique random `id` that helps you identify the Document. It can be used to {ref}`access this Document inside a DocumentArray<access-elements>`.

````{tip}
When you `print()` a Document, you get a string representation such as `<Document ('id', 'tensor') at a14f01a66d3e11ec8bde1e008a366d49>`. It shows the non-empty attributes of that Document as well as its `id`, which helps you understand the content of that Document.

```text
<Document ('id', 'tensor') at a14f01a66d3e11ec8bde1e008a366d49>
           ^^^^^^^^^^^^^^     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                 |                          |
                 |                          |
          non-empty fields                  |
                                      Document.id
```
````

You can learn more about constructing new Documents in the {ref}`Construct chapter<construct-doc>`.

You can also construct Documents from bytes, JSON, and Protobuf messages. These methods are introduced in the {ref}`Serialization chapter<serialize>`.

One of the most powerful features of Documents is their ability to hold nested data. This is explained further in the {ref}`Dataclass section<dataclass>`.

## Constructing DocumentArrays

A DocumentArray is a list-like container of Document objects. To create an empty DocumentArray:

```python
from docarray import Document, DocumentArray

da = DocumentArray()
```

```text
<DocumentArray (length=0) at 4453362704>
```

Now you can use list-like interfaces such as `.append()` and `.extend()` as you would to add elements to a Python list—in fact, DocumentArray implements **all** list interfaces. This means that if you know how to use a Python `list`, you already know how to use a DocumentArray.

```python
da.append(Document(text='hello world!'))
da.extend([Document(text='hello'), Document(text='world!')])
```

```text
<DocumentArray (length=3) at 4446140816>
```

Directly printing a DocumentArray doesn't show much useful information. Instead, you can use {meth}`~docarray.array.mixins.plot.PlotMixin.summary`.


```python
da.summary()
```

```text
                  Documents Summary                   
                                                      
  Type                   DocumentArrayInMemory
  Length                 3                            
  Homogenous Documents   True                         
  Common Attributes      ('id', 'text')  
  Multimodal dataclass   False
                                                      
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    3                False            
  text        ('str',)    3                False    
```

## Serializing DocumentArrays

You can serialize your DocumentArray in a variety of ways:

```python
da.to_json()

da.save_binary('my_docarray.bin', protocol='protobuf', compress='lz4')

da.to_dataframe()
```

These and other serialization formats are detailed in the {ref}`Serialization chapter<docarray-serialization>`.

## Interaction with Jina AI Cloud

```{important}
This feature requires the `rich` and `requests` dependencies. You can do `pip install "docarray[full]"` to install them.
```

The {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.push` and {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.pull` methods allow you to serialize a DocumentArray object to Jina AI Cloud and share it across machines.

```python
from docarray import DocumentArray

da = DocumentArray(...)  # heavy lifting, processing, GPU tasks...
da.push('myda123', show_progress=True)
```

```{figure} ../documentarray/images/da-push.png

```

Then on your local laptop, simply pull it:

```python
from docarray import DocumentArray

da = DocumentArray.pull('myda123', show_progress=True)
```

Further details are available in the {ref}`Interaction with Jina AI Cloud chapter<interaction-cloud>`.