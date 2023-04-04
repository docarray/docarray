# Video

!!! note
    This requires a `av` dependency. You can install all necessary dependencies via:
    ```cmd 
    pip install "docarray[video]"
    ```

## Load video data

![type:video](mov_bbb.mp4){: style='width: 600px; height: 330px'}

```python hl_lines="15"
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
doc.video, doc.audio, doc.key_frame_indices = doc.url.load()

print(type(vid.video), vid.video.shape)

```
```text
<class 'docarray.typing.tensor.video.video_ndarray.VideoNdArray'> 
(250, 176, 320, 3)
```

Video data is represented as a video tensor, an audio tensor and an array key frame indices. 
The video tensor is a 4-dim array with shape=(n_frames, height, width, channels). The first dimension represents the frame id. 
The last three dimensions represent the same thing as in image data. 
In the given example with vid.video.shape=(250, 176, 320, 3), the video contains 250 frames of size 176x320. 
Based on the overall length of the video (10s), we can infer the framerate is around 250/10=25fps.
If the video contains audio, it will be stored as an AudioNdArray in vid.audio.
Additionally, the key frame indices will be stored. A key frame is defined as the starting point of any smooth transition.

## Key frame extraction
A key frame is defined as the starting point of any smooth transition.

With the key frame indices you acn easily access selected scenes:
```python
indices = vid.key_frame_indices
first_scene = vid.video[indices[0] : indices[1]]
print(indices)
print(first_scene.shape)
```
```text
[0, 95]
(95, 176, 320, 3)
```

Or you can easily access the first frame of all new scenes. 
```python
key_frames = vid.video[vid.key_frame_indices]
print(key_frames.shape)
```
```text
(2, 176, 320, 3)
```
To display them, cast them to ImageNdArrays:
```python
from pydantic import parse_obj_as

for frame in key_frames:
    img = parse_obj_as(ImageNdArray, frame)
    img.display()
```


<figure markdown>
  ![](key_frames.png){ width="400" }
</figure>



## Save video to file

You can easily save your video tensor to a file. In this example we save the video with a framerate of 60, which results in a 4 sec video, instead of the original 10 second video with frame rate 25. and optionally hand over the corresponding audio tensor:
```python
vid.video.save(
    file_path="/path/my_audio.mp4",
    video_frame_rate=60,
)
```


![type:video](mov_bbb_framerate_60.mp4){: style='width: 600px; height: 330px'}


## Display video 

You can display a video from a VideoUrl or a VideoNdArray. For the latter you can optionally give the corresponding AudioTensor as a parameter:

## Predefined VideoDoc

