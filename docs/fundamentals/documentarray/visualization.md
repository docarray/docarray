# Visualization

## Summary in table

You are already familiar with {meth}`~docarray.array.mixins.plot.PlotMixin.summary`, which prints a summary table for a DocumentArray and its attributes:

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

If a DocumentArray contains only image Documents, you can plot them all in one sprite image using {meth}`~docarray.array.mixins.plot.PlotMixin.plot_image_sprites`.

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

If an image Document contains images in its `.matches` attribute, you can visualise the matching results using {meth}`~docarray.document.mixins.plot.PlotMixin.plot_matches_sprites`.

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
This feature requires `fastapi` dependency. You can run `pip install "docarray[full]"` to install it.
```

If a DocumentArray has `.embeddings`, you can visualize them interactively using {meth}`~docarray.array.mixins.plot.PlotMixin.plot_embeddings`.

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

For an image DocumentArray, you can pass the `image_sprites` parameter to set the visualization points to images.

```python
da.plot_embeddings(image_sprites=True)
```

```{figure} images/embedding-projector.gif
:align: center
```

````{admonition} Note
:class: note
If you have a lot of metadata, plotting may be slow since that metadata is stored in a corresponding TSV file. You can speed up plotting with the `exclude_fields_metas` parameter, preventing fields (like `chunks` or `matches`) from being written to the TSV.
````
