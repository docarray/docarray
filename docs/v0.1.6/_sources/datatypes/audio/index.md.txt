(audio-type)=
# {octicon}`unmute` Audio

## Load `.wav` file 

To load a wav file as a Document.

```python
from docarray import Document

d = Document(uri='toy.wav').load_uri_to_audio_blob()

print(d.blob.shape, d.blob.dtype)
```

```text
(30833,) float32
```

## Save as `.wav` file

You can save Document `.blob` as a `.wav` file:

```python
d.save_audio_blob_to_file('toy.wav')
```


## Example

Let's load the "hello" audio file, reverse it and finally save it.

```python
from docarray import Document

d = Document(uri='hello.wav').load_uri_to_audio_blob()
d.blob = d.blob[::-1]
d.save_audio_blob_to_file('olleh.wav')
```

<table>
  <tr>
    <th>hello.wav</th>
    <th>olleh.wav</th>
  </tr>
  <tr>
    <td><audio controls><source src="../../_static/hello.wav" type="audio/wav"></audio></td>
    <td><audio controls><source src="../../_static/olleh.wav" type="audio/wav"></audio></td>
  </tr>
</table>


## Other tools & libraries for audio data

By no means you are restricted to use DocArray native methods for audio processing. Here are some command-line tools, programs and libraries to use for more advanced handling of audio data:

- [`FFmpeg`](https://ffmpeg.org) is a free, open-source project for handling multimedia files and streams. 
- [`pydub`](https://github.com/jiaaro/pydub): manipulate audio with a simple and easy high level interface
- [`librosa`](https://librosa.github.io/librosa/): a python package for music and audio analysis.
- [`pyAudioAnalysis`](https://github.com/tyiannak/pyAudioAnalysis): for IO or for more advanced feature extraction and signal analysis.

