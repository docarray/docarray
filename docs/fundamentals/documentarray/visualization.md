# Visualization

## Summary in table

We are already pretty familiar with {meth}`~docarray.array.mixins.plot.PlotMixin.summary`, which prints a table of summary for DocumentArray and its attributes:

```python
from docarray import DocumentArray

da = DocumentArray.empty(3)

da.summary()
```

```text
        Documents Summary         
                                  
  Length                 3        
  Homogenous Documents   True     
  Common Attributes      ('id',)  
                                  
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    3                False            
```

## Image sprites

If a DocumentArray contains all image Documents, you can plot all images in one sprite image using {meth}`~docarray.array.mixins.plot.PlotMixin.plot_image_sprites`.

```python
from docarray import DocumentArray

docs = DocumentArray.from_files('*.jpg')
docs.plot_image_sprites()
```

```{figure} images/sprite-image.png
:width: 60%
```
(plot-matches)=
### Plot Matches

If an image Document contains the matching images in its `.matches` attribute, you can visualise the matching results using {meth}`~docarray.document.mixins.plot.PlotMixin.plot_matches_sprites`.

```python
import numpy as np
from docarray import DocumentArray

da = DocumentArray.from_files('*.jpg')
da.embeddings = np.random.random([len(da), 10])
da.match(da)
da[0].plot_matches_sprites(top_k=5, channel_axis=-1, inv_normalize=False)
```

```{figure} images/sprite-match.png
:width: 60%
```

(visualize-embeddings)=
## Embedding projector

```{important}
This feature requires `fastapi` dependency. You can do `pip install "docarray[full]"` to install it.
```

If a DocumentArray has `.embeddings`, you can visualize the embeddings interactively using {meth}`~docarray.array.mixins.plot.PlotMixin.plot_embeddings`.

```python
import numpy as np
from docarray import DocumentArray

docs = DocumentArray.empty(1000)
docs.embeddings = np.random.random([len(docs), 256])

docs.plot_embeddings()
```

```{figure} images/embedding-projector-empty.gif
:align: center
```

For image DocumentArray, you can do one step more to attach the image sprite on to the visualization points.

```python
da.plot_embeddings(image_sprites=True)
```

```{figure} images/embedding-projector.gif
:align: center
```