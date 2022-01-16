(image-type)=
# {octicon}`image` Image

````{tip}
This requires `Pillow` and `matplotlib` dependencies. You can install them via `pip install "docarray[full]"`
````


## Load image data

You can load image data by specifying the image URI and then convert it into `.tensor` using Document API

```{figure} apple.png
:align: center
:scale: 30%
```

```python
from docarray import Document

d = Document(uri='apple.png')
d.load_uri_to_image_tensor()

print(d.tensor, d.tensor.shape)
```

```text
[[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
  [255 255 255]]]
(618, 641, 3)
```

## Simple image processing

DocArray provides some functions to help you preprocess the image data. You can resize it (i.e. downsampling/upsampling) and normalize it; you can switch the channel axis of the `.tensor` to meet certain requirements of other framework; and finally you can chain all these preprocessing steps together in one line. For example, before feeding data into a Pytorch-based ResNet Executor, the image needs to be normalized and the color axis should be at first, not at the last. You can do this via:

```python
from docarray import Document

d = (
    Document(uri='apple.png')
    .load_uri_to_image_tensor()
    .set_image_tensor_shape(shape=(224, 224))
    .set_image_tensor_normalization()
    .set_image_tensor_channel_axis(-1, 0)
)

print(d.tensor, d.tensor.shape)
```


```text
[[[2.2489083 2.2489083 2.2489083 ... 2.2489083 2.2489083 2.2489083]
  [2.2489083 2.2489083 2.2489083 ... 2.2489083 2.2489083 2.2489083]
  [2.2489083 2.2489083 2.2489083 ... 2.2489083 2.2489083 2.2489083]
  ...
  [2.64      2.64      2.64      ... 2.64      2.64      2.64     ]
  [2.64      2.64      2.64      ... 2.64      2.64      2.64     ]
  [2.64      2.64      2.64      ... 2.64      2.64      2.64     ]]]
(3, 224, 224)
```

You can also dump `.tensor` back to a PNG image so that you can see.

```python
d.save_image_tensor_to_file('apple-proc.png', channel_axis=0)
```

Note that the channel axis is now switched to 0 because the previous preprocessing steps we just conducted. 

```{figure} apple-proc.png
:align: center
:scale: 30%
```

Yep, this looks uneatable. That's often what you give to the deep learning algorithms. 

## Display image sprite

An image sprites is a collection of images put into a single image. When working with a DocumentArray of image Documents, you can directly view the image sprites via `plot_image_sprites`. This gives you a quick view of the dataset that you are working with:

```python
from docarray import DocumentArray

da = DocumentArray.from_files('/Users/hanxiao/Downloads/left/*.jpg')
da.plot_image_sprites('sprite-img.png')
```

Depending on the number of images, this could take a while. But after that, you get a very nice overview of your DocumentArray as follows:

```{figure} sprite-img.png
:align: center
:width: 70%
```

## Segment large complicated image into small ones

A large complicated image is hard to search, as it may contain too many elements and interesting information and hence hard to define the search problem in the first place. Take the following image as an example, 

```{figure} complicated-image.jpeg
:align: center
:width: 80%
```

It contains rich information in details, and it is complicated as there is no single salience interest in the image. The user may want to hit this image by searching for "Krusty Burger" or "Yellow schoolbus". User's real intention is hard guess, which highly depends on the applications. But at least what we can do is using DocArray to breakdown this complicated image into simpler ones. One of the simplest approaches is to cut the image via sliding windows.

```python
from docarray import Document

d = Document(uri='docs/datatype/image/complicated-image.jpeg')
d.load_uri_to_image_tensor()
print(d.tensor.shape)

d.convert_image_tensor_to_sliding_windows(window_shape=(64, 64))
print(d.tensor.shape)
```

```text
(792, 1000, 3)
(180, 64, 64, 3)
```

As one can see, it converts the single image tensor into 180 image tensors, each with the size of (64, 64, 3). You can also add all 180 image tensors into the chunks of this `Document`, simply do:

```python
d.convert_image_tensor_to_sliding_windows(window_shape=(64, 64), as_chunks=True)

print(d.chunks)
```

```text
ChunkArray has 180 items (showing first three):
{'id': '7585b8aa-3826-11ec-bc1a-1e008a366d48', 'mime_type': 'image/jpeg', 'tensor': {'dense': {'buffer': 'H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0H8T0 ...
```

Let's now use image sprite to see how these chunks look like:

```python
d.chunks.plot_image_sprites('simpsons-chunks.png')
```

```{figure} simpsons-chunks.png
:align: center
:width: 80%
```

Hmm, doesn't change so much. This is because we scan the whole image using sliding windows with no overlap (i.e. stride). Let's do a bit oversampling:

```python
d.convert_image_tensor_to_sliding_windows(window_shape=(64, 64), strides=(10, 10), as_chunks=True)
d.chunks.plot_image_sprites('simpsons-chunks-stride-10.png')
```

```{figure} simpsons-chunks-stride.png
:align: center
:width: 80%
```

Yep, that definitely looks better.

