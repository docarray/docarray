# 🎥 Video

DocArray supports many modalities including `Video`.
This section will show you how to load and handle video data using DocArray.

Moreover, you will learn about DocArray's video specific types, to represent your video data ranging from [`VideoUrl`][docarray.typing.url.VideoUrl] to [`VideoBytes`][docarray.typing.bytes.VideoBytes] and [`VideoNdArray`][docarray.typing.tensor.video.video_ndarray.VideoNdArray].

!!! note
    This requires a `av` dependency. You can install all necessary dependencies via:
    ```cmd 
    pip install "docarray[video]"
    ```

## Load video data

In DocArray video data is represented by a video tensor, an audio tensor and the key frame indices. 

![type:video](mov_bbb.mp4){: style='width: 600px; height: 330px'}

First let's define a `MyVideo` class with all of those attributes and instantiate an object with a local or remote url:

```python
from docarray import BaseDoc
from docarray.typing import AudioNdArray, NdArray, VideoNdArray, VideoUrl


class MyVideo(BaseDoc):
    url: VideoUrl
    video: VideoNdArray = None
    audio: AudioNdArray = None
    key_frame_indices: NdArray = None


doc = MyVideo(
    url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
)
```

Now you can load the video file content by simply calling [`.load()`][docarray.typing.url.audio_url.AudioUrl.load] on your [`AudioUrl`][docarray.typing.url.audio_url.AudioUrl] instance.
This will return a [NamedTuple](https://docs.python.org/3/library/typing.html#typing.NamedTuple) of a **video tensor**, an **audio tensor** and the **key frame indices**:

- The video tensor is a 4-dim array of shape `(n_frames, height, width, channels)`. <br>The first dimension represents the frame id. 
The last three dimensions represent the image data of the corresponding frame. 

- If the video contains audio, it will be stored as an AudioNdArray.

- Additionally, the key frame indices will be stored. A key frame is defined as the starting point of any smooth transition.


```python
doc.video, doc.audio, doc.key_frame_indices = doc.url.load()

assert isinstance(doc.video, VideoNdArray)
assert isinstance(doc.audio, AudioNdArray)
assert isinstance(doc.key_frame_indices, NdArray)

print(doc.video.shape)
```
``` { .text .no-copy }
(250, 176, 320, 3)
```
For the given example you can infer from `doc.video`'s shape, that the video contains 250 frames of size 176x320 in RGB mode. 
Based on the overall length of the video (10s), you can infer the framerate is approximately 250/10 = 25 frames per second (fps).


## VideoTensor

DocArray offers several [`VideoTensor`'s](../../../../api_references/typing/tensor/video) to store your data to:

- [`VideoNdArray`][docarray.typing.tensor.video.video_ndarray.VideoNdArray]
- [`VideoTorchTensor`][docarray.typing.tensor.video.VideoTorchTensor]
- [`VideoTensorFlowTensor`][docarray.typing.tensor.video.VideoTensorFlowTensor]

If you specify the type of your tensor to one of the above, it will be cast to that automatically:

```python hl_lines="7 8 15 16"
from docarray import BaseDoc
from docarray.typing import VideoTensorFlowTensor, VideoTorchTensor, VideoUrl


class MyVideo(BaseDoc):
    url: VideoUrl
    tf_tensor: VideoTensorFlowTensor = None
    torch_tensor: VideoTorchTensor = None


doc = MyVideo(
    url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
)

doc.tf_tensor = doc.url.load().video
doc.torch_tensor = doc.url.load().video

assert isinstance(doc.tf_tensor, VideoTensorFlowTensor)
assert isinstance(doc.torch_tensor, VideoTorchTensor)
```



## VideoBytes

Alternatively, you can load your [`VideoUrl`][docarray.typing.url.VideoUrl] instance to [`VideoBytes`][docarray.typing.bytes.VideoBytes], and your [`VideoBytes`][docarray.typing.bytes.VideoBytes] instance to a [`VideoTensor`](../../../../api_references/typing/tensor/video) of your choice:

```python hl_lines="15 16"
from docarray import BaseDoc
from docarray.typing import VideoTensor, VideoUrl, VideoBytes


class MyVideo(BaseDoc):
    url: VideoUrl
    bytes_: VideoBytes = None
    video: VideoTensor = None


doc = MyVideo(
    url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
)

doc.bytes_ = doc.url.load_bytes()
doc.video = doc.url.load().video
```
 
Vice versa, you can also transform a [`VideoTensor`](../../../../api_references/typing/tensor/video) to  [`VideoBytes`][docarray.typing.bytes.VideoBytes]:

```python
from docarray.typing import VideoBytes

bytes_from_tensor = doc.video.to_bytes()

# assert isinstance(bytes_from_tensor, VideoBytes)
```


## Key frame extraction

A key frame is defined as the starting point of any smooth transition.
Given the key frame indices you can access selected scenes.

```python
indices = doc.key_frame_indices
first_scene = doc.video[indices[0] : indices[1]]

assert (indices == [0, 95]).all()
assert first_scene.shape == (95, 176, 320, 3)
```

Or you can access the first frame of all new scenes and display them in a notebook:

```python
from docarray.typing import ImageNdArray
from pydantic import parse_obj_as


key_frames = doc.video[doc.key_frame_indices]
for frame in key_frames:
    img = parse_obj_as(ImageNdArray, frame)
    img.display()
```

<figure markdown>
  ![](key_frames.png){ width="350" }
</figure>



## Save video to file

You can save your video tensor to a file. In the example below you save the video with a framerate of 60 fps, which results in a 4 sec video, instead of the original 10 second video with a frame rate of 25 fps. 
```python
doc.video.save(
    file_path="/path/my_video.mp4",
    video_frame_rate=60,
)
```

## Display video in notebook

You can play a video in a notebook from its URL as well as its tensor, by calling `.display()` on either one. For the latter you can optionally give the corresponding [`AudioTensor`](../../../../api_references/typing/tensor/audio) as a parameter.

```python
doc_fast = MyAudio(url="/path/my_video.mp4")
doc_fast.url.display()
```
![type:video](mov_bbb_framerate_60.mp4){: style='width: 600px; height: 330px'}



## Getting started - Predefined `VideoDoc`

To get started and play around with your video data, DocArray provides a predefined [`VideoDoc`][docarray.documents.video.VideoDoc], which includes all of the previously mentioned functionalities:

```python
class VideoDoc(BaseDoc):
    url: Optional[VideoUrl]
    audio: Optional[AudioDoc] = AudioDoc()
    tensor: Optional[VideoTensor]
    key_frame_indices: Optional[AnyTensor]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[bytes]
```

You can use this class directly or extend it to your preference:

```python
from typing import Optional

from docarray.documents import VideoDoc


# extend it
class MyVideo(VideoDoc):
    name: Optional[str]


video = MyVideo(
    url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
)
video.name = 'My first video doc!'
video.video_tensor = video.url.load().video
model = MyEmbeddingModel()
video.embedding = model(video.tensor)
```