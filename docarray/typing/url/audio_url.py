import wave
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T', bound='AudioUrl')


class AudioUrl(AnyUrl):
    """
    URL to a .wav file.
    Can be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that needs to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(audio_url=str(self))

    def load(self: T) -> np.ndarray:
        """
        Load the data from the url into a numpy.ndarray audio tensor.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document
            import numpy as np

            from docarray.typing import AudioUrl


            class MyDoc(Document):
                audio_url: AudioUrl


            doc = MyDoc(mesh_url="toydata/hello.wav")

            audio_tensor = doc.audio_url.load()
            assert isinstance(audio_tensor, np.ndarray)

        :return: np.ndarray representing the audio file content
        """

        if self.startswith('http'):
            import io

            import requests

            resp = requests.get(self)
            resp.raise_for_status()
            file = io.BytesIO()
            file.write(resp.content)
            file.seek(0)
        else:
            file = self

        # note wave is Python built-in mod. https://docs.python.org/3/library/wave.html
        with wave.open(file) as ifile:
            samples = ifile.getnframes()
            audio = ifile.readframes(samples)

            # Convert buffer to float32 using NumPy
            audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
            audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

            # Normalise float32 array so that values are between -1.0 and +1.0
            max_int16 = 2**15
            audio_normalised = audio_as_np_float32 / max_int16

            channels = ifile.getnchannels()
            if channels == 2:
                # 1 for mono, 2 for stereo
                audio_stereo = np.empty(
                    (int(len(audio_normalised) / channels), channels)
                )
                audio_stereo[:, 0] = audio_normalised[
                    range(0, len(audio_normalised), 2)
                ]
                audio_stereo[:, 1] = audio_normalised[
                    range(1, len(audio_normalised), 2)
                ]

                return audio_stereo
            else:
                return audio_normalised
