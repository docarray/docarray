# Image

````{tip}
This requires `Pillow` dependencies. You can install them via `pip install "docarray[image]"`
````
DocArray offers several Image specific types to represent your image data starting with an image url to image bytes to an image tensor.

## Load data
You can load image data by specifying the image url as an `ImageUrl` and then call .load() on it.

You can store your image data in an ImageTensor, which can be an:
- ImageNdArray
- ImageTorchTensor
- ImageTensorFlowTensor.

By default, Loading the image data from your ImageUrl instance returns an ImageNdArray instance. 

```python
from docarray.typing import ImageTensorFlowTensor, ImageTensor, ImageUrl
from docarray import BaseDoc


class MyImage(BaseDoc):
    url: ImageUrl = None
    tensor: ImageTensor = None
    tf_tensor: ImageTensorFlowTensor = None


img = MyImage(url='apple.png')

img.tensor = img.url.load()
img.tf_tensor = img.url.load()

print(type(img.tensor), type(img.tf_tensor))
```
```text
<class 'docarray.typing.tensor.image.image_tensorflow_tensor.ImageTensorFlowTensor'>
<class 'docarray.typing.tensor.image.image_torch_tensor.ImageTorchTensor'>
```
The load() method by default outputs an ImageNdArray. If you specify the type of your tensor to ImageTensorFlowTensor or ImageTorchTensor, it will be cast to that automatically:

## ImageBytes

You can also load your image data into ImageBytes and the load the tensor data from the ImageBytes instance:
```python
from docarray.typing import ImageBytes, ImageTensor, ImageUrl
from docarray import BaseDoc


class MyImage(BaseDoc):
    url: ImageUrl = None
    bytes_: ImageBytes = None
    tensor: ImageTensor = None


img = MyImage(url='apple.png')

img.bytes_ = img.url.load_bytes()
img.tensor = img.bytes_.load()

print(f"type(img.bytes_) = {type(img.bytes_)}")
print(f"type(img.tensor) = {type(img.tensor)}")
```
```
type(img.bytes_) = <class 'docarray.typing.bytes.image_bytes.ImageBytes'>
type(img.tensor) = <class 'docarray.typing.tensor.image.image_ndarray.ImageNdArray'>
```

## Display image in notebook

You can display your image data in a notebook from both an url as well as a tensor

```{figure} display_notebook.jpeg
:align: center
:width: 80%
```

## Predefined ImageDoc

To get started and play around with the image modality we provide a predefined ImageDoc, which includes all of the previously mentioned functionalities:
```python
class ImageDoc(BaseDoc):
    url: Optional[ImageUrl]
    tensor: Optional[ImageTensor]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[ImageBytes]
```

You can use this class directly:


Or extend it: