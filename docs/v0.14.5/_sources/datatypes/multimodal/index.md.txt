(multimodal-example)=
# {octicon}`stack` Multi-modal

This example will walk you through how to use DocArray to process multiple data modalities, jointly.
To do this comfortably and cleanly, you will use DocArray's {ref}`dataclass <dataclass>` feature.

```{seealso}
This example works with image and text data.
If you are not yet familiar with how to process these modalities individually, you may want to check out the
respective examples first: {ref}`Image <image-type>` and {ref}`Text <text-type>`
```

## Model your data

If you work with multiple modalities at the same time, most likely they stand in some relation with each other.
DocArray's dataclass feature allows you to model your data and these relationships, using the language of your domain.

Suppose you want to model a page of a newspaper that contains a main text, an image, and an image description.
You can model this example in the following way:

```python
from docarray import dataclass
from docarray.typing import Image, Text


@dataclass
class Page:
    main_text: Text
    image: Image
    description: Text
```

### Instantiate a Document

After defining the data model through dataclasses, you can instantiate the dataclasses with your actual data, and cast it to a {class}`~docarray.document.Document`:


```python
from docarray import Document

page = Page(
    main_text='Hello world',
    image='apple.png',
    description='This is the image of an apple',
)

doc = Document(page)
```
Finally, you can see the nested Document structure that was created automatically:

```python
doc.summary()
```

````{dropdown} Output

```text
📄 Document: 7f03e397da8725aa8a2aed4a0d64f263
└── 💠 Chunks
    ├── 📄 Document: 627c3b052b86e908b10475a4649ce49b
    │   ╭──────────────────────┬────────────────────────────────────────────────
    │   │ Attribute            │ Value                                          
    │   ├──────────────────────┼────────────────────────────────────────────────
    │   │ parent_id            │ 7f03e397da8725aa8a2aed4a0d64f263               
    │   │ granularity          │ 1                                              
    │   │ text                 │ Hello world                                    
    │   │ modality             │ text                                           
    │   ╰──────────────────────┴────────────────────────────────────────────────
    ├── 📄 Document: 79e75c074aa444341baac18549930450
    │   ╭──────────────┬────────────────────────────────────────────────────────
    │   │ Attribute    │ Value                                                  
    │   ├──────────────┼────────────────────────────────────────────────────────
    │   │ parent_id    │ 7f03e397da8725aa8a2aed4a0d64f263                       
    │   │ granularity  │ 1                                                      
    │   │ tensor       │ <class 'numpy.ndarray'> in shape (618, 641, 3), dtype: 
    │   │ mime_type    │ image/png                                              
    │   │ uri          │ apple.png                                              
    │   │ modality     │ image                                                  
    │   ╰──────────────┴────────────────────────────────────────────────────────
    └── 📄 Document: 6861a1e3d77c3560a630dee34ba5ac7f
        ╭──────────────────────┬────────────────────────────────────────────────
        │ Attribute            │ Value                                          
        ├──────────────────────┼────────────────────────────────────────────────
        │ parent_id            │ 7f03e397da8725aa8a2aed4a0d64f263               
        │ granularity          │ 1                                              
        │ text                 │ This is the image of an apple                  
        │ modality             │ text                                           
        ╰──────────────────────┴────────────────────────────────────────────────

```

````

### Nested dataclasses and list types

If your domain requires a more complex model, you can use advanced features to represent that accurately.

For this example, we look at a journal which consists of a cover page and multiple other pages, as well as some metadata.
Further, each page contains a main text, and can contain and image and an image description.

```python
from docarray import dataclass
from docarray.typing import Image, Text, JSON
from typing import List


@dataclass
class Page:
    main_text: Text
    image: Image = None
    description: Text = None


@dataclass
class Journal:
    cover: Page
    pages: List[Page]
    metadata: JSON
```

You can instantiate this complex Document in the same way as before:

```python
from docarray import Document

pages = [
    Page(
        main_text='Hello world',
        image='apple.png',
        description='This is the image of an apple',
    ),
    Page(main_text='Second page'),
]

journal = Journal(
    cover=Page(main_text='DocArray Daily', image='apple.png'),
    pages=pages,
    metadata={'author': 'Jina AI', 'issue': '1'},
)

doc = Document(journal)
doc.summary()
```

````{dropdown} Output

```text
📄 Document: cab4e047bc84ffb6b8b0597ff4ee0e9f
└── 💠 Chunks
    ├── 📄 Document: ea686d21029e4687df83a6ee31af98b2
    │   ╭──────────────────────┬────────────────────────────────────────────────
    │   │ Attribute            │ Value                                          
    │   ├──────────────────────┼────────────────────────────────────────────────
    │   │ parent_id            │ cab4e047bc84ffb6b8b0597ff4ee0e9f               
    │   │ granularity          │ 1                                              
    │   ╰──────────────────────┴────────────────────────────────────────────────
    │   └── 💠 Chunks
    │       ├── 📄 Document: 139a5f16ab176b5c9d5088b1f2792973
    │       │   ╭──────────────────────┬────────────────────────────────────────
    │       │   │ Attribute            │ Value                                  
    │       │   ├──────────────────────┼────────────────────────────────────────
    │       │   │ parent_id            │ ea686d21029e4687df83a6ee31af98b2       
    │       │   │ granularity          │ 1                                      
    │       │   │ text                 │ DocArray Daily                         
    │       │   │ modality             │ text                                   
    │       │   ╰──────────────────────┴────────────────────────────────────────
    │       └── 📄 Document: f1e7527757c7dc6006fa8fa36e7b788f
    │           ╭──────────────┬────────────────────────────────────────────────
    │           │ Attribute    │ Value                                          
    │           ├──────────────┼────────────────────────────────────────────────
    │           │ parent_id    │ ea686d21029e4687df83a6ee31af98b2               
    │           │ granularity  │ 1                                              
    │           │ tensor       │ <class 'numpy.ndarray'> in shape (618, 641, 3),
    │           │ mime_type    │ image/png                                      
    │           │ uri          │ apple.png                                      
    │           │ modality     │ image                                          
    │           ╰──────────────┴────────────────────────────────────────────────
    ├── 📄 Document: 2a13aee3a2ac8eadc07f43bc2dd83583
    │   ╭──────────────────────┬────────────────────────────────────────────────
    │   │ Attribute            │ Value                                          
    │   ├──────────────────────┼────────────────────────────────────────────────
    │   │ parent_id            │ cab4e047bc84ffb6b8b0597ff4ee0e9f               
    │   │ granularity          │ 1                                              
    │   ╰──────────────────────┴────────────────────────────────────────────────
    │   └── 💠 Chunks
    │       ├── 📄 Document: b6bcfa7000a25bd84ddcd35813c99b4c
    │       │   ╭──────────────────────┬────────────────────────────────────────
    │       │   │ Attribute            │ Value                                  
    │       │   ├──────────────────────┼────────────────────────────────────────
    │       │   │ parent_id            │ 2a13aee3a2ac8eadc07f43bc2dd83583       
    │       │   │ granularity          │ 1                                      
    │       │   ╰──────────────────────┴────────────────────────────────────────
    │       │   └── 💠 Chunks
    │       │       ├── 📄 Document: 71018fd73c13187309590e82b5255416
    │       │       │   ╭──────────────────────┬────────────────────────────────
    │       │       │   │ Attribute            │ Value                          
    │       │       │   ├──────────────────────┼────────────────────────────────
    │       │       │   │ parent_id            │ b6bcfa7000a25bd84ddcd35813c99b4
    │       │       │   │ granularity          │ 1                              
    │       │       │   │ text                 │ Hello world                    
    │       │       │   │ modality             │ text                           
    │       │       │   ╰──────────────────────┴────────────────────────────────
    │       │       ├── 📄 Document: b335f748006204dd27bb5fa9a99a572f
    │       │       │   ╭──────────────┬────────────────────────────────────────
    │       │       │   │ Attribute    │ Value                                  
    │       │       │   ├──────────────┼────────────────────────────────────────
    │       │       │   │ parent_id    │ b6bcfa7000a25bd84ddcd35813c99b4c       
    │       │       │   │ granularity  │ 1                                      
    │       │       │   │ tensor       │ <class 'numpy.ndarray'> in shape (618, 
    │       │       │   │ mime_type    │ image/png                              
    │       │       │   │ uri          │ apple.png                              
    │       │       │   │ modality     │ image                                  
    │       │       │   ╰──────────────┴────────────────────────────────────────
    │       │       └── 📄 Document: 7769657ae7c25227920b5ae35a2a3c31
    │       │           ╭──────────────────────┬────────────────────────────────
    │       │           │ Attribute            │ Value                          
    │       │           ├──────────────────────┼────────────────────────────────
    │       │           │ parent_id            │ b6bcfa7000a25bd84ddcd35813c99b4
    │       │           │ granularity          │ 1                              
    │       │           │ text                 │ This is the image of an apple  
    │       │           │ modality             │ text                           
    │       │           ╰──────────────────────┴────────────────────────────────
    │       └── 📄 Document: 29f1835bac77e435f00976c5cf4e97cb
    │           ╭──────────────────────┬────────────────────────────────────────
    │           │ Attribute            │ Value                                  
    │           ├──────────────────────┼────────────────────────────────────────
    │           │ parent_id            │ 2a13aee3a2ac8eadc07f43bc2dd83583       
    │           │ granularity          │ 1                                      
    │           ╰──────────────────────┴────────────────────────────────────────
    │           └── 💠 Chunks
    │               └── 📄 Document: bc8adb52bad51ccff3d6e7834a4b536a
    │                   ╭──────────────────────┬────────────────────────────────
    │                   │ Attribute            │ Value                          
    │                   ├──────────────────────┼────────────────────────────────
    │                   │ parent_id            │ 29f1835bac77e435f00976c5cf4e97c
    │                   │ granularity          │ 1                              
    │                   │ text                 │ Second page                    
    │                   │ modality             │ text                           
    │                   ╰──────────────────────┴────────────────────────────────
    └── 📄 Document: c602af33ed3f2d693a5633e53b87e19c
        ╭─────────────────────┬─────────────────────────────────────────────────
        │ Attribute           │ Value                                           
        ├─────────────────────┼─────────────────────────────────────────────────
        │ parent_id           │ cab4e047bc84ffb6b8b0597ff4ee0e9f                
        │ granularity         │ 1                                               
        │ tags                │ {'author': 'Jina AI', 'issue': '1'}             
        │ modality            │ json                                            
        ╰─────────────────────┴─────────────────────────────────────────────────

```

````

## Access the data

After instantiation, each modality can be accessed directly from the Document:

```python
from docarray import dataclass, Document
from docarray.typing import Image, Text


@dataclass
class Page:
    main_text: Text
    image: Image
    description: Text


page = Page(
    main_text='Hello world',
    image='apple.png',
    description='This is the image of an apple',
)

doc = Document(page)

print(doc.main_text)
print(doc.main_text.text)
print(doc.image)
print(doc.image.tensor)
```

```text
<Document ('id', 'parent_id', 'granularity', 'text', 'modality') at 1ee83d2c391f078736732bb34a021587>
Hello world
<Document ('id', 'parent_id', 'granularity', 'tensor', 'mime_type', 'uri', '_metadata', 'modality') at c8fe3b8fd101bea6a4820a53d2993bdf>
[[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
  [255 255 255]
  [255 255 255]
  [255 255 255]]]

```

## Generate embeddings

Common use cases, such as neural search, involve generating embeddings for your data.

There are two ways of doing this, each of which has its use cases:
Generating individually embeddings for each modality, and generating an overall embedding for the entire Document.

### Embed each modality

If you want to create an embedding for each modality of each page, you can simply access the corresponding Document, and add an embedding vector.

This can be useful, for example, when you want to compare different Documents based on a specific modality that they store.
```python
from torchvision.models import resnet50

img_model = resnet50(pretrained=True)

# embed textual data
doc.main_text.embed_feature_hashing()
doc.description.embed_feature_hashing()
# embed image data
doc.image.set_image_tensor_shape(shape=(224, 224)).set_image_tensor_channel_axis(
    original_channel_axis=-1, new_channel_axis=0
).set_image_tensor_normalization(channel_axis=0).embed(img_model)

print(doc.main_text.embedding.shape)
print(doc.description.embedding.shape)
print(doc.image.embedding.shape)
```

```text
(256,)
(256,)
torch.Size([1000])

```

If you have a {class}`~docarray.array.document.DocumentArray` of multi-modal Documents, you can embed the modalities of each
Document in the following way:

```python
from docarray import DocumentArray, Document

da = DocumentArray(
    [
        Document(
            Page(
                main_text='First page',
                image='apple.png',
                description='This is the image of an apple',
            )
        ),
        Document(
            Page(
                main_text='Second page',
                image='apple.png',
                description='Still the same image of the same apple',
            )
        ),
    ]
)

from torchvision.models import resnet50

img_model = resnet50(pretrained=True)

# embed textual data
da['@.[description, main_text]'].apply(lambda d: d.embed_feature_hashing())
# embed image data
da['@.[image]'].apply(
    lambda d: d.set_image_tensor_shape(shape=(224, 224))
    .set_image_tensor_channel_axis(original_channel_axis=-1, new_channel_axis=0)
    .set_image_tensor_normalization(channel_axis=0)
)
da['@.[image]'].embed(img_model)

print(da['@.[description, main_text]'].embeddings.shape)
print(da['@.[image]'].embeddings.shape)
```

```text
(4, 256)
torch.Size([2, 1000])

```


### Embed parent Document

From the individual embeddings you can create a combined embedding for the entire Document.
This can be useful, for example, when you want to compare different Documents based on all the modalities that they store.

```python
import numpy as np


def combine_embeddings(d):
    # any (more sophisticated) function could go here
    d.embedding = np.concatenate(
        [d.image.embedding, d.main_text.embedding, d.description.embedding]
    )
    return d


da.apply(combine_embeddings)
print(da.embeddings.shape)
```

```text
(2, 1512)
```

## Perform search

With the embeddings from above you can tackle downstream tasks, such as neural search.

### Find Document by modality embedding

Let's assume you have multiple pages, and you want to find the page that contains a similar image as some other page
(the query page).

First, create your dataset and query Document:

```python
from docarray import dataclass, Document, DocumentArray
from docarray.typing import Image, Text


@dataclass
class Page:
    main_text: Text
    image: Image
    description: Text


query_page = Page(
    main_text='Hello world',
    image='apple.png',
    description='This is the image of an apple',
)

query = Document(query_page)  # our query Document

da = DocumentArray(
    [
        Document(
            Page(
                main_text='First page',
                image='apple.png',
                description='This is the image of an apple',
            )
        ),
        Document(
            Page(
                main_text='Second page',
                image='pear.png',
                description='This is an image of a pear',
            )
        ),
    ]
)  # our dataset of pages
```

Then you can embed your dataset and your query:

```python
from torchvision.models import resnet50

img_model = resnet50(pretrained=True)

# embed query
query.image.set_image_tensor_shape(shape=(224, 224)).set_image_tensor_channel_axis(
    original_channel_axis=-1, new_channel_axis=0
).set_image_tensor_normalization(channel_axis=0).embed(img_model)

# embed dataset
da['@.[image]'].apply(
    lambda d: d.set_image_tensor_shape(shape=(224, 224))
    .set_image_tensor_channel_axis(original_channel_axis=-1, new_channel_axis=0)
    .set_image_tensor_normalization(channel_axis=0)
).embed(img_model)
```

Finally, cou can perform a search using {meth}`~docarray.array.document.DocumentArray.find` to find the closest image,
and the parent Document that contains that image:

```python
closest_match_img = da['@.[image]'].find(query.image)[0][0]
print('CLOSEST IMAGE:')
closest_match_img.summary()
print('PAGE WITH THE CLOSEST IMAGE:')
closest_match_page = da[closest_match_img.parent_id]
closest_match_page.summary()
```

````{dropdown} Output
```text
CLOSEST IMAGE:
📄 Document: 5922ee1ad0dbfe707301b573f98c5939
╭─────────────┬────────────────────────────────────────────────────────────────╮
│ Attribute   │ Value                                                          │
├─────────────┼────────────────────────────────────────────────────────────────┤
│ parent_id   │ e6266f88f6ebcb3417358440934bcf81                               │
│ granularity │ 1                                                              │
│ tensor      │ <class 'numpy.ndarray'> in shape (3, 224, 224), dtype: float32 │
│ mime_type   │ image/png                                                      │
│ uri         │ apple.png                                                      │
│ embedding   │ <class 'torch.Tensor'> in shape (1000,), dtype: float32        │
│ modality    │ image                                                          │
│ scores      │ defaultdict(<class 'docarray.score.NamedScore'>, {'cosine':    │
│             │ {'value': -1.1920929e-07}})                                    │
╰─────────────┴────────────────────────────────────────────────────────────────╯
PAGE WITH THE CLOSEST IMAGE:
📄 Document: e6266f88f6ebcb3417358440934bcf81
└── 💠 Chunks
    ├── 📄 Document: 29a0e323e2e9befcc42e9823b111f90f
    │   ╭──────────────────────┬────────────────────────────────────────────────
    │   │ Attribute            │ Value                                          
    │   ├──────────────────────┼────────────────────────────────────────────────
    │   │ parent_id            │ e6266f88f6ebcb3417358440934bcf81               
    │   │ granularity          │ 1                                              
    │   │ text                 │ First page                                     
    │   │ modality             │ text                                           
    │   ╰──────────────────────┴────────────────────────────────────────────────
    ├── 📄 Document: 5922ee1ad0dbfe707301b573f98c5939
    │   ╭─────────────┬─────────────────────────────────────────────────────────
    │   │ Attribute   │ Value                                                   
    │   ├─────────────┼─────────────────────────────────────────────────────────
    │   │ parent_id   │ e6266f88f6ebcb3417358440934bcf81                        
    │   │ granularity │ 1                                                       
    │   │ tensor      │ <class 'numpy.ndarray'> in shape (3, 224, 224), dtype: f
    │   │ mime_type   │ image/png                                               
    │   │ uri         │ apple.png                                               
    │   │ embedding   │ <class 'torch.Tensor'> in shape (1000,), dtype: float32 
    │   │ modality    │ image                                                   
    │   ╰─────────────┴─────────────────────────────────────────────────────────
    └── 📄 Document: 175e386b1aa248f9387db46341b73e05
        ╭──────────────────────┬────────────────────────────────────────────────
        │ Attribute            │ Value                                          
        ├──────────────────────┼────────────────────────────────────────────────
        │ parent_id            │ e6266f88f6ebcb3417358440934bcf81               
        │ granularity          │ 1                                              
        │ text                 │ This is the image of an apple                  
        │ modality             │ text                                           
        ╰──────────────────────┴────────────────────────────────────────────────

```
````


### Find Document by combined embedding

Similarly, you might want to find the page that is *overall* most similar to the other pages in your dataset.

To do that, you first have to embed each modality for each Document, and then combine the embeddings to an overall embedding:

```python
from torchvision.models import resnet50
import numpy as np

img_model = resnet50(pretrained=True)

# embed text data in query and dataset
query.main_text.embed_feature_hashing()
query.description.embed_feature_hashing()
da['@.[description, main_text]'].apply(lambda d: d.embed_feature_hashing())

# combine embeddings to overall embedding
def combine_embeddings(d):
    # any (more sophisticated) function could go here
    d.embedding = np.concatenate(
        [d.image.embedding, d.main_text.embedding, d.description.embedding]
    )
    return d


query = combine_embeddings(query)  # combine embeddings for query
da.apply(combine_embeddings)  # combine embeddings in dataset
```

Then, you can perform search directly on the top level:

```python
closest_match_page = da.find(query)[0][0]
print('OVERALL CLOSEST PAGE:')
closest_match_page.summary()
```

````{dropdown} Output
```text
OVERALL CLOSEST PAGE:
📄 Document: a0f33de91bb7d53811c7cb3015cdf1b8
╭───────────┬──────────────────────────────────────────────────────────────────╮
│ Attribute │ Value                                                            │
├───────────┼──────────────────────────────────────────────────────────────────┤
│ embedding │ <class 'numpy.ndarray'> in shape (1512,), dtype: float64         │
│ scores    │ defaultdict(<class 'docarray.score.NamedScore'>, {'cosine':      │
│           │ {'value': 0.01911603045476573}})                                 │
╰───────────┴──────────────────────────────────────────────────────────────────╯
└── 💠 Chunks
    ├── 📄 Document: b9c9206422e27c8a0f1b4a4e745901ec
    │   ╭─────────────┬─────────────────────────────────────────────────────────
    │   │ Attribute   │ Value                                                   
    │   ├─────────────┼─────────────────────────────────────────────────────────
    │   │ parent_id   │ a0f33de91bb7d53811c7cb3015cdf1b8                        
    │   │ granularity │ 1                                                       
    │   │ text        │ First page                                              
    │   │ embedding   │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    │   │             │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    │   │             │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    │   │             │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    │   │             │ ▄▄▄▄▄▄▄                                                 
    │   │ modality    │ text                                                    
    │   ╰─────────────┴─────────────────────────────────────────────────────────
    ├── 📄 Document: 97cbca49aeaf0ea1f609b161dc9ec934
    │   ╭─────────────┬─────────────────────────────────────────────────────────
    │   │ Attribute   │ Value                                                   
    │   ├─────────────┼─────────────────────────────────────────────────────────
    │   │ parent_id   │ a0f33de91bb7d53811c7cb3015cdf1b8                        
    │   │ granularity │ 1                                                       
    │   │ tensor      │ <class 'numpy.ndarray'> in shape (3, 224, 224), dtype: f
    │   │ mime_type   │ image/png                                               
    │   │ uri         │ apple.png                                               
    │   │ embedding   │ <class 'torch.Tensor'> in shape (1000,), dtype: float32 
    │   │ modality    │ image                                                   
    │   ╰─────────────┴─────────────────────────────────────────────────────────
    └── 📄 Document: 9183813fc38f291f353b76d4125506d6
        ╭─────────────┬─────────────────────────────────────────────────────────
        │ Attribute   │ Value                                                   
        ├─────────────┼─────────────────────────────────────────────────────────
        │ parent_id   │ a0f33de91bb7d53811c7cb3015cdf1b8                        
        │ granularity │ 1                                                       
        │ text        │ This is the image of an apple                           
        │ embedding   │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        │             │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        │             │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        │             │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        │             │ ▄▄▄▄▄▄▄                                                 
        │ modality    │ text                                                    
        ╰─────────────┴─────────────────────────────────────────────────────────

```
````