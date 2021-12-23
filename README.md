# docarray

<!-- start elevator-pitch -->

The data structure for unstructured data.

🌌 **All data types**: super-expressive data structure for representing complicated/mixed/nested text, image, video, audio, 3D mesh data.

🧑‍🔬 **Data science powerhouse**: easy-to-use functions for facilitating data scientists work on embedding, matching, visualizing, evaluating unstructured data via Torch/Tensorflow/ONNX/PaddlePaddle.

🚡 **Portable**: ready to wire with efficient conversion from/to Protobuf, binary, JSON, CSV, dataframe.

<!-- end elevator-pitch -->


## Install 

Requires Python 3.7+ and `numpy`:
```
pip install docarray
```
To install full dependencies, please use `pip install docarray[full]`.

## [Documentation](https://docarray.jina.ai)

## Get Started

Let's use DocArray and ResNet50 to build a meme image search on [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset). This dataset contains 6016 image-pairs stored in `/left` and `/right`. Images that shares the same filename are labeled as perceptually similar. For example, 

| `/left` | `/right` | `/left` | `/right` |
|---------|----------|---------|----------|
|

Our problem is given an image from `/left` and find its most-similar image in `/right` (without looking at the filename of course).

### Load images

First load images and preprocess them with standard computer vision techniques:

```python
from docarray import DocumentArray, Document

left_da = DocumentArray.from_files('left/*.jpg')
```

To get a feeling of the data you will handle, plot them in one sprite image:

```python
left_da.plot_image_sprites()
```

<p align="center">
<a href="https://docs.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/master/.github/README-img/sprite.png?raw=true" alt="Load totally looks like dataset with docarray API" width="70%"></a>
</p>

### Apply preprocess

Let's do some standard computer vision preprocessing:

```python
def preproc(d: Document):
    return (d.load_uri_to_image_blob()  # load
             .set_image_blob_normalization()  # normalize color 
             .set_image_blob_channel_axis(-1, 0))  # switch color axis

left_da.apply(preproc)
```

Did I mention `apply` work in parallel? Never mind. 

### Embed images

Now convert images into embeddings using a pretrained ResNet50:

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
left_da.embed(model, device='cuda')  # embed via GPU to speedup
```

### Visualize embeddings

You can visualize the embeddings via tSNE in an interactive embedding projector:

```python
left_da.plot_embeddings()
```

<p align="center">
<a href="https://docs.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/master/.github/README-img/tsne.gif?raw=true" alt="Visualizing embedding via tSNE and embedding projector" width="90%"></a>
</p>

Fun is fun, but recall our goal is to match left images against right images and so far we have only handled the left. Let's repeat the same procedure for the right:

```python
right_da = (DocumentArray.from_files('right/*.jpg')
                         .apply(preproc)
                         .embed(model, device='cuda'))
```

### Match nearest neighbours

We can now match the left to the right.

```python
left_da.match(right_da, limit=10)
```

Let's inspect what's inside `left_da` now:

```python
for d in left_da:
    for m in d.matches:
        print(d.uri, m.uri, m.scores['cosine'].value)
```

```text
left/02262.jpg right/03459.jpg 0.21102
left/02262.jpg right/02964.jpg 0.13871843
left/02262.jpg right/02103.jpg 0.18265384
left/02262.jpg right/04520.jpg 0.16477376
...
```

Better see it with eyes.

```python

```

### Quantitative evaluation


