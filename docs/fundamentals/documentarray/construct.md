(construct-array)=
# Construct

## Construct an empty array

```python
from docarray import DocumentArray

da = DocumentArray()
```

```text
<DocumentArray (length=0) at 4453362704>
```

Now you can use list-like interfaces such as `.append()` and `.extend()` as you would to add elements to a Python List.

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
                                                      
  Length                 3                            
  Homogenous Documents   True                         
  Common Attributes      ('id', 'mime_type', 'text')  
                                                      
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    3                False            
  mime_type   ('str',)    1                False            
  text        ('str',)    3                False    
```

## Construct with empty Documents

You can quickly build a DocumentArray with only empty Documents, similar to `numpy.zeros()`:

```python
from docarray import DocumentArray

da = DocumentArray.empty(10)
```

```text
<DocumentArray (length=10) at 4453362704>
```

## Construct from list-like objects

You can construct a DocumentArray from a `Sequence`, `List`, `Tuple`, or an `Iterator` that yields `Document` objects.

````{tab} From list of Documents
```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])
```

```text
<DocumentArray (length=2) at 4866772176>
```

````
````{tab} From generator
```python
from docarray import DocumentArray, Document

da = DocumentArray((Document() for _ in range(10)))
```

```text
<DocumentArray (length=10) at 4866772176>
```
````


As DocumentArray itself is also a "list-like object that yields `Document` objects", you can also construct a DocumentArray from another DocumentArray:

```python
da = DocumentArray(...)
da1 = DocumentArray(da)
```


## Construct from multiple DocumentArrays

You can use `+` or `+=` to concatenate DocumentArrays together:

```python
from docarray import DocumentArray

da1 = DocumentArray.empty(3)
da2 = DocumentArray.empty(4)
da3 = DocumentArray.empty(5)
print(da1 + da2 + da3)

da1 += da2
print(da1)
```

```text
<DocumentArray (length=12) at 5024988176>
<DocumentArray (length=7) at 4525853328>
```


## Construct from a single Document

```python
from docarray import DocumentArray, Document

d1 = Document(text='hello')
da = DocumentArray(d1)
```

```text
<DocumentArray (length=1) at 4452802192>
```

## Deep copy on elements

Note that, as in Python list, adding a Document object into DocumentArray only adds its memory reference. The original Document is *not* copied. If you change the original Document afterwards, then the one inside the DocumentArray will also change. Here is an example:

```python
from docarray import DocumentArray, Document

d1 = Document(text='hello')
da = DocumentArray(d1)

print(da[0].text)
d1.text = 'world'
print(da[0].text)
```

```text
hello
world
```

This may surprise some users, but considering the following Python code, you will find this behavior is very natural and authentic.

```python
d = {'hello': None}
a = [d]

print(a[0]['hello'])
d['hello'] = 'world'
print(a[0]['hello'])
```

```text
None
world
```

To make a deep copy, set `DocumentArray(..., copy=True)`. Now all Documents in this DocumentArray are completely new objects with identical contents as the original ones.

```python
from docarray import DocumentArray, Document

d1 = Document(text='hello')
da = DocumentArray(d1, copy=True)

print(da[0].text)
d1.text = 'world'
print(da[0].text)
```

```text
hello
hello
```

## Construct from local files

You may recall the common pattern that {ref}`I mentioned here<content-uri>`. With {meth}`~docarray.document.generators.from_files`, one can easily construct a DocumentArray object with all file paths defined by a glob expression. 

```python
from docarray import DocumentArray

da_jpg = DocumentArray.from_files('images/*.jpg')
da_png = DocumentArray.from_files('images/*.png')
da_all = DocumentArray.from_files(['images/**/*.png', 'images/**/*.jpg', 'images/**/*.jpeg'])
```

This will scan all filenames that match the expression and construct Documents with filled `.uri` attributes. You can specify whether to read each as text or binary with the `read_mode` argument.




## What's next?

In the next chapter, we will see how to construct DocumentArrays from binary bytes, JSON, CSV, dataframe, and Protobuf message.