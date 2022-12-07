(bulk-access)=
# Access Attributes

DocumentArray itself has no attribute. Accessing attributes in this context means access attributes of the contained Documents in bulk.

In the last chapter, we get a taste of the powerful element selector of the DocumentArray. This chapter will continue talking about the attribute selector.


## Attribute selector

```python
da[element_selector, attribute_selector]
```

Here `element_selector` are the ones introduced {ref}`in the last chapter<access-elements>`. The attribute selector can be a string, or a list/tuple of string that represents the names of the attributes.

As in element selector, one can use attribute selector to **get/set/delete** attributes in a DocumentArray.

| Example                                      | Return                                                                                                                        |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `da[:, 'id']`                                | all `.id` from all root Documents in a List                                                                                   |
| `da[..., 'id']`                              | all `.id` from all flattened Documents (root, chunks, and matches) in a List                                                                                |
| `da['@m', 'id']`                             | all `.id` from all Documents `.matches`                                                                                       |
| `da[1:3, ('id', 'scores')]`                  | a list of two list, first is all `.id` from the first three Documents, second is all `.scores` from the first three Documents |
| `da[1:3, 'scores__cosine__value']`             | all `.scores['cosine'].value` from the first three Documents                                                                  |
| `da[1:3, 'embedding']`, `da[1:3].embeddings` | a NdArray-like object of the first three Documents embeddings                                                                 |
| `da[:, 'tensor']`, `da.tensors`              | a NdArray-like object of the all top-level Documents tensors                                                                  |


Let's see an example.

```python
from docarray import DocumentArray

da = DocumentArray().empty(3)
for d in da:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

print(da[:, 'id'])
```

```text
['8d41ce5c6f0d11eca2181e008a366d49', '8d41cfa66f0d11eca2181e008a366d49', '8d41cff66f0d11eca2181e008a366d49']
```

Of course you can use it with {ref}`the path-string selector<path-string>`.

```python
print(da['@c', 'id'])
```

```text
['db60ab8a6f0d11ec99511e008a366d49', 'db60abda6f0d11ec99511e008a366d49', 'db60c12e6f0d11ec99511e008a366d49', 'db60c1886f0d11ec99511e008a366d49', 'db60c4266f0d11ec99511e008a366d49', 'db60c46c6f0d11ec99511e008a366d49']
```

```python
print(da[..., 'id'])
```

```text
['285db6586f0e11ec99401e008a366d49', '285db6b26f0e11ec99401e008a366d49', '285dbff46f0e11ec99401e008a366d49', '285dc0586f0e11ec99401e008a366d49', '285db3606f0e11ec99401e008a366d49', '285dcc746f0e11ec99401e008a366d49', '285dccce6f0e11ec99401e008a366d49', '285dce0e6f0e11ec99401e008a366d49', '285dce5e6f0e11ec99401e008a366d49', '285db4fa6f0e11ec99401e008a366d49', '285dcf946f0e11ec99401e008a366d49', '285dcfda6f0e11ec99401e008a366d49', '285dd1066f0e11ec99401e008a366d49', '285dd16a6f0e11ec99401e008a366d49', '285db55e6f0e11ec99401e008a366d49']
```

Let's set the field `mime_type` for top-level Documents. We have three top-level Documents, so:

```python
da[:, 'mime_type'] = ['image/jpg', 'image/png', 'image/jpg']

da.summary()
```

```text
                          Documents Summary                           
                                                                      
  Length                    3                                         
  Homogenous Documents      True                                      
  Has nested Documents in   ('chunks', 'matches')                     
  Common Attributes         ('id', 'mime_type', 'chunks', 'matches')  
                                                                      
                        Attributes Summary                        
                                                                  
  Attribute   Data type         #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────────── 
  chunks      ('ChunkArray',)   3                False            
  id          ('str',)          3                False            
  matches     ('MatchArray',)   3                False            
  mime_type   ('str',)          2                False            
```

We can see `mime_type` are set.

If you want to set an attribute of all Documents to the same value without looping:

```python
da[:, 'mime_type'] = 'hello'
```

One can also select multiple attributes in one-shot:

```python
da[:, ['mime_type', 'id']]
```

```text
[['image/jpg', 'image/png', 'image/jpg'], ['095cd76a6f0f11ec82211e008a366d49', '095cd8d26f0f11ec82211e008a366d49', '095cd92c6f0f11ec82211e008a366d49']]
```

Now let's remove them.

```python
del da[:, 'mime_type']

da.summary()
```

```text                                                                  
                    Documents Summary                    
                                                         
  Length                    3                            
  Homogenous Documents      True                         
  Has nested Documents in   ('chunks', 'matches')        
  Common Attributes         ('id', 'chunks', 'matches')  
                                                         
                        Attributes Summary                        
                                                                  
  Attribute   Data type         #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────────── 
  chunks      ('ChunkArray',)   3                False            
  id          ('str',)          3                False            
  matches     ('MatchArray',)   3                False            
                                                                  
```

## Auto-ravel on NdArray

Attribute selectors `tensor` and `embedding` behave a bit differently. Instead of relying on Python List for input & return when get/set, they automatically ravel/unravel the NdArray-like object [^1] for you.

[^1]: NdArray-like can be Numpy/TensorFlow/PyTorch/SciPy/PaddlePaddle sparse & dense array.

Here is an example, where one may expect that `da[:, 'embedding']` gives you a list of three `(1, 10)` COO matrices. But it auto ravels the results and returns as a `(3, 10)` COO matrix:

```python
import numpy as np
import scipy.sparse
from docarray import DocumentArray

# build sparse matrix
sp_embed = np.random.random([3, 10])
sp_embed[sp_embed > 0.1] = 0
sp_embed = scipy.sparse.coo_matrix(sp_embed)

da = DocumentArray.empty(3)

da[:, 'embedding'] = sp_embed

print(type(da[:, 'embedding']), da[:, 'embedding'].shape)
for d in da:
    print(type(d.embedding), d.embedding.shape)
```

```text
<class 'scipy.sparse.coo.coo_matrix'> (3, 10)
<class 'scipy.sparse.coo.coo_matrix'> (1, 10)
<class 'scipy.sparse.coo.coo_matrix'> (1, 10)
<class 'scipy.sparse.coo.coo_matrix'> (1, 10)
```

Auto unravel works in a similar way, we just assign a `(3, 10)` COO matrix as `.embeddings` and it auto breaks into three and assign them into the three Documents. 

Of course, this is not limited to scipy sparse matrix. Any NdArray-like[^1] object would work. The same logic applies also to `.tensors` attribute.

## Dunder syntax for nested attributes

Some attributes are nested by nature, e.g. `.tags` and `.scores`. Accessing the deep nested value is easy thanks to the dunder (double under) expression. You can access `.tags['key1']` via `d[:, 'tags__key1']`. 

Let's see an example,

```python
import numpy as np

from docarray import DocumentArray

da = DocumentArray.empty(3)
da.embeddings = np.random.random([3, 2])
da.match(da)
```

Now to print `id` and matched score, one can simply do:

```python
print(da['@m', ('id', 'scores__cosine__value')])
```

```text
[['5164d792709a11ec9ae71e008a366d49', '5164d986709a11ec9ae71e008a366d49', '5164d922709a11ec9ae71e008a366d49', '5164d922709a11ec9ae71e008a366d49', '5164d986709a11ec9ae71e008a366d49', '5164d792709a11ec9ae71e008a366d49', '5164d986709a11ec9ae71e008a366d49', '5164d792709a11ec9ae71e008a366d49', '5164d922709a11ec9ae71e008a366d49'], 
[0.0, 0.006942970007385196, 0.48303283924326845, 0.0, 0.3859268166910603, 0.48303283924326845, 2.220446049250313e-16, 0.006942970007385196, 0.3859268166910603]]
```


(da-content-embedding)=
## Content and embedding sugary attributes

DocumentArray provides `.texts`, `.blobs`, `.tensors`, `.contents` and `.embeddings` sugary attributes for quickly accessing the content and embedding of Documents. You can use them to get/set/delete attributes of all Documents at the top-level.

```python
from docarray import DocumentArray

da = DocumentArray.empty(2)
da.texts = ['hello', 'world']

print(da.texts)
```

```text
['hello', 'world']
```

This is same as `da[:, 'text'] = ['hello', 'world']` and then `print(da[:, 'text'])` but more compact and probably more Pythonic.

Same for `.tensors` and `.embeddings`:

```python
import numpy as np
from docarray import DocumentArray

# build sparse matrix
embed = np.random.random([3, 10])

da = DocumentArray.empty(3)

da.embeddings = embed

print(type(da.embeddings), da.embeddings.shape)
for d in da:
    print(type(d.embedding), d.embedding.shape)
```

```text
<class 'numpy.ndarray'> (3, 10)
<class 'numpy.ndarray'> (10,)
<class 'numpy.ndarray'> (10,)
<class 'numpy.ndarray'> (10,)
```
