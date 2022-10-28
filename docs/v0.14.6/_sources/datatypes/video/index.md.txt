(video-type)=
# {octicon}`device-camera-video` Video


````{tip}

This feature requires `av` dependency. You can install it via `pip install "docarray[full]"`.

````


## Load video data


<video controls width="60%">
<source src="../../_static/mov_bbb.mp4" type="video/mp4">
</video>


```python
from docarray import Document

d = Document(uri='toy.mp4')
d.load_uri_to_video_tensor()

print(d.tensor.shape)
```

```text
(250, 176, 320, 3)
```

For video data, `.tensor` is a 4-dim array, where the first dimension represents the frame id, or time. The last three dimensions represent the same thing as in image data. Here we got our `d.tensor.shape=(250, 176, 320, 3)`, which means this video is sized in 176x320 and contains 250 frames. Based on the overall length of the video (10s), we can infer the framerate is around 250/10=25fps.

We can put each frame into a sub-Document in `.chunks` as use image sprite to visualize them.

```python
for b in d.tensor:
    d.chunks.append(Document(tensor=b))

d.chunks.plot_image_sprites('mov.png')
```

```{figure} mov_bbb.png
:align: center
:width: 70%
```

## Key frame extraction

From the sprite image one can observe our example video is quite redundant. Let's extract the key frames from this video and see:

```python
from docarray import Document

d = Document(uri='toy.mp4')
d.load_uri_to_video_tensor(only_keyframes=True)
print(d.tensor.shape)
```

```text
(2, 176, 320, 3)
```

Looks like we only have two key frames, let's dump them into images and see what do they look like.

```python
for idx, c in enumerate(d.tensor):
    Document(tensor=c).save_image_tensor_to_file(f'chunk-{idx}.png')
```

```{figure} chunk-0.png
:align: center
:width: 40%
```

```{figure} chunk-1.png
:align: center
:width: 40%
```

Makes sense, right?

## Save as video file

One can also save a Document `.tensor` as a video file. In this example, we load our `.mp4` video and store it into a 60fps video.

```python
from docarray import Document

d = (
    Document(uri='toy.mp4')
    .load_uri_to_video_tensor()
    .save_video_tensor_to_file('60fps.mp4', 60)
)
```

<video controls width="60%">
<source src="../../_static/60fps.mp4" type="video/mp4">
</video>

