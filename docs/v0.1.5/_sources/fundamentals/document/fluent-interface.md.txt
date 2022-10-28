# Fluent Interface

Document provides a simple fluent interface that allows one to process (often preprocess) a Document object by chaining methods. For example to read an image file as `numpy.ndarray`, resize it, normalize it and then store it to another file; one can simply do:

```python
from docarray import Document

d = (
    Document(uri='apple.png')
        .load_uri_to_image_blob()
        .set_image_blob_shape((64, 64))
        .set_image_blob_normalization()
        .save_image_blob_to_file('apple1.png')
)
```

```{figure} images/apple.png
:scale: 20%

Original `apple.png`
```

```{figure} images/apple1.png
:scale: 50%

Processed `apple1.png`
```


Note that, chaining methods always modify the original Document in-place. That means the above example is equivalent to:

```python
from docarray import Document

d = Document(uri='apple.png')

(d.load_uri_to_image_blob()
  .set_image_blob_shape((64, 64))
  .set_image_blob_normalization()
  .save_image_blob_to_file('apple1.png'))
```


## Methods

All the following methods can be chained.


<!-- fluent-interface-start -->
### Convert
Provide helper functions for {class}`Document` to support conversion between {attr}`.blob`, {attr}`.text`
and {attr}`.buffer`.
- {meth}`~docarray.document.mixins.convert.ConvertMixin.convert_blob_to_buffer`
- {meth}`~docarray.document.mixins.convert.ConvertMixin.convert_buffer_to_blob`
- {meth}`~docarray.document.mixins.convert.ConvertMixin.convert_uri_to_datauri`


### TextData
Provide helper functions for {class}`Document` to support text data.
- {meth}`~docarray.document.mixins.text.TextDataMixin.convert_blob_to_text`
- {meth}`~docarray.document.mixins.text.TextDataMixin.convert_text_to_blob`
- {meth}`~docarray.document.mixins.text.TextDataMixin.convert_text_to_datauri`
- {meth}`~docarray.document.mixins.text.TextDataMixin.load_uri_to_text`


### BufferData
Provide helper functions for {class}`Document` to handle binary data.
- {meth}`~docarray.document.mixins.buffer.BufferDataMixin.convert_buffer_to_datauri`
- {meth}`~docarray.document.mixins.buffer.BufferDataMixin.load_uri_to_buffer`
- {meth}`~docarray.document.mixins.buffer.BufferDataMixin.save_buffer_to_file`


### ImageData
Provide helper functions for {class}`Document` to support image data.
- {meth}`~docarray.document.mixins.image.ImageDataMixin.convert_buffer_to_image_blob`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.convert_image_blob_to_buffer`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.convert_image_blob_to_sliding_windows`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.convert_image_blob_to_uri`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.load_uri_to_image_blob`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.save_image_blob_to_file`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.set_image_blob_channel_axis`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.set_image_blob_inv_normalization`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.set_image_blob_normalization`
- {meth}`~docarray.document.mixins.image.ImageDataMixin.set_image_blob_shape`


### ContentProperty
Provide helper functions for {class}`Document` to allow universal content property access.
- {meth}`~docarray.document.mixins.content.ContentPropertyMixin.convert_content_to_datauri`


### SingletonSugar
Provide sugary syntax for {class}`Document` by inheriting methods from {class}`DocumentArray`
- {meth}`~docarray.document.mixins.sugar.SingletonSugarMixin.embed`
- {meth}`~docarray.document.mixins.sugar.SingletonSugarMixin.match`


### FeatureHash
Provide helper functions for feature hashing.
- {meth}`~docarray.document.mixins.featurehash.FeatureHashMixin.embed_feature_hashing`


### Porting

- {meth}`~docarray.document.mixins.porting.PortingMixin.from_bytes`
- {meth}`~docarray.document.mixins.porting.PortingMixin.from_dict`
- {meth}`~docarray.document.mixins.porting.PortingMixin.from_json`


### Protobuf

- {meth}`~docarray.document.mixins.protobuf.ProtobufMixin.from_protobuf`


### AudioData
Provide helper functions for {class}`Document` to support audio data.
- {meth}`~docarray.document.mixins.audio.AudioDataMixin.load_uri_to_audio_blob`
- {meth}`~docarray.document.mixins.audio.AudioDataMixin.save_audio_blob_to_file`


### MeshData
Provide helper functions for {class}`Document` to support 3D mesh data and point cloud.
- {meth}`~docarray.document.mixins.mesh.MeshDataMixin.load_uri_to_point_cloud_blob`


### VideoData
Provide helper functions for {class}`Document` to support video data.
- {meth}`~docarray.document.mixins.video.VideoDataMixin.load_uri_to_video_blob`
- {meth}`~docarray.document.mixins.video.VideoDataMixin.save_video_blob_to_file`


### UriFile
Provide helper functions for {class}`Document` to dump content to a file.
- {meth}`~docarray.document.mixins.dump.UriFileMixin.save_uri_to_file`


<!-- fluent-interface-end -->
