# Video


````{tip}
This requires a `av` dependency. You can install it via `pip install "docarray[video]"`
````

## Load video data

1
<video markdown controls="">
    <source src="../mov_bbb.mp4" type="video/mp4">
</video>

2
<figure markdown>
  ![type:video](mov_bbb.mp4){ width="900" type="video/mp4"}
</figure>

2b
<figure markdown>
  ![type:video](mov_bbb.mp4){ width="900"}
</figure>

3
<figure markdown>
  ![type:video](../mov_bbb.mp4){ width="900" }
</figure>

3b
<figure markdown>
  ![type:video](../mov_bbb.mp4){ width="900" type="video/mp4"}
</figure>

4
![type:video](mov_bbb.mp4){: style='width: 300'}

5
<video markdown controls="">
    <source src="mov_bbb.mp4" type="video/mp4">
</video>


8
<iframe width="420" height="315" 
    src="../mov_bbb.mp4">
</iframe>

```python
from docarray import BaseDoc
from docarray.typing import VideoUrl


class MyVideo(BaseDoc):
    url: VideoUrl


doc = MyVideo(
    url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
)
vid = doc.url.load()

print(type(vid.video), vid.video.shape)
print(type(vid.audio), vid.audio.shape)
print(type(vid.key_frame_indices), vid.key_frame_indices.shape)
```
```text
<class 'docarray.typing.tensor.video.video_ndarray.VideoNdArray'> 
<class 'docarray.typing.tensor.audio.audio_ndarray.AudioNdArray'> 
<class 'docarray.typing.tensor.ndarray.NdArray'>
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

<video width="300" controls>
    <source src="mov_bbb_framerate_60.mp4" type="video/mp4">
</video>



## Display video 

You can display a video from a VideoUrl or a VideoNdArray. For the latter you can optionally give the corresponding AudioTensor as a parameter:

## Predefined VideoDoc

