# Audio


!!! note
    This requires a `pydub` dependency. You can install it via `pip install "docarray[audio]"`


## Load audio file
You can store the url of any audio file as an AudioUrl instance. Loading the content of the audio file is as easy as calling `.load()` on this AudioUrl instance. This will give you a numpy.ndarray representing the audio file content and an integer of the corresponding frame rate, which describes the number of signals for a certain period of time.

```python
from docarray import BaseDoc
from docarray.typing import AudioUrl, AudioNdArray


class MyAudio(BaseDoc):
    url: AudioUrl
    tensor: AudioNdArray = None
    frame_rate: int = None


doc = MyAudio(url='https://www.kozco.com/tech/piano2.wav')
doc.tensor, doc.frame_rate = doc.url.load()

doc.summary()
```
```text
📄 MyAudio : 8b05512 ...
╭──────────────────────┬───────────────────────────────────────────────────────╮
│ Attribute            │ Value                                                 │
├──────────────────────┼───────────────────────────────────────────────────────┤
│ url: AudioUrl        │ https://www.kozco.com/tech/piano2.wav                 │
│ tensor: AudioNdArray │ AudioNdArray of shape (605424,), dtype: float64       │
│ frame_rate: int      │ 48000                                                 │
╰──────────────────────┴───────────────────────────────────────────────────────╯
```

## Save audio to file
You can save your AudioTensor to an audio file of any format as follows:
```python
doc.tensor.save(
    file_path='path/my_audio.mp3',
    format='mp3',
    frame_rate=doc.frame_rate,
)
```
## Play audio in notebook

You can play your audio sound in a notebook from its url as well as its tensor, by calling `.display()` on either one:

```python
doc.url.display()
```

<table>
  <tr>
    <th>hello.wav</th>
  </tr>
  <tr>
    <td><audio controls><source src="https://www.kozco.com/tech/piano2.wav" type="audio/wav"></audio></td>
  </tr>
</table>

## Predefined AudioDoc
