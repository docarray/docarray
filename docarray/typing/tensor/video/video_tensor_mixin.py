import abc
import warnings
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.audio.audio_tensor import AudioTensor
from docarray.utils._internal.misc import import_library, is_notebook

if TYPE_CHECKING:
    from docarray.typing.bytes.video_bytes import VideoBytes

T = TypeVar('T', bound='VideoTensorMixin')


class VideoTensorMixin(AbstractTensor, abc.ABC):
    @classmethod
    def validate_shape(cls: Type['T'], value: 'T') -> 'T':
        comp_be = cls.get_comp_backend()
        shape = comp_be.shape(value)  # type: ignore
        if comp_be.n_dim(value) not in [3, 4] or shape[-1] != 3:  # type: ignore
            raise ValueError(
                f'Expects tensor with 3 or 4 dimensions and the last dimension equal '
                f'to 3, but received {shape}.'
            )
        else:
            return value

    def save(
        self: 'T',
        file_path: Union[str, BytesIO],
        audio_tensor: Optional[AudioTensor] = None,
        video_frame_rate: int = 24,
        video_codec: str = 'h264',
        audio_frame_rate: int = 48000,
        audio_codec: str = 'aac',
        audio_format: str = 'fltp',
    ) -> None:
        """
        Save video tensor to a .mp4 file.

        ---

        ```python
        import numpy as np

        from docarray import BaseDoc
        from docarray.typing.tensor.audio.audio_tensor import AudioTensor
        from docarray.typing.tensor.video.video_tensor import VideoTensor


        class MyDoc(BaseDoc):
            video_tensor: VideoTensor
            audio_tensor: AudioTensor


        doc = MyDoc(
            video_tensor=np.random.randint(low=0, high=256, size=(10, 200, 300, 3)),
            audio_tensor=np.random.randn(100, 1, 1024).astype("float32"),
        )

        doc.video_tensor.save(
            file_path="/tmp/mp_.mp4",
            audio_tensor=doc.audio_tensor,
            audio_format="flt",
        )
        ```

        ---
        :param file_path: path to a .mp4 file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param audio_tensor: AudioTensor containing the video's soundtrack.
        :param video_frame_rate: video frames per second.
        :param video_codec: the name of a video decoder/encoder.
        :param audio_frame_rate: audio frames per second.
        :param audio_codec: the name of an audio decoder/encoder.
        :param audio_format: the name of one of the audio formats supported by PyAV,
            such as 'flt', 'fltp', 's16' or 's16p'.
        """
        if TYPE_CHECKING:
            import av
        else:
            av = import_library('av', raise_error=True)

        np_tensor = self.get_comp_backend().to_numpy(array=self)
        video_tensor = np_tensor.astype('uint8')

        if isinstance(file_path, str):
            format = file_path.split('.')[-1]
        else:
            format = 'mp4'

        with av.open(file_path, mode='w', format=format) as container:
            if video_tensor.ndim == 3:
                video_tensor = np.expand_dims(video_tensor, axis=0)

            stream_video = container.add_stream(video_codec, rate=video_frame_rate)
            stream_video.height = video_tensor.shape[-3]
            stream_video.width = video_tensor.shape[-2]

            if audio_tensor is not None:
                stream_audio = container.add_stream(audio_codec)
                audio_np = audio_tensor.get_comp_backend().to_numpy(array=audio_tensor)
                audio_layout = 'stereo' if audio_np.shape[-2] == 2 else 'mono'

                for i, audio in enumerate(audio_np):
                    frame = av.AudioFrame.from_ndarray(
                        array=audio, format=audio_format, layout=audio_layout
                    )
                    frame.rate = audio_frame_rate
                    frame.pts = audio.shape[-1] * i
                    for packet in stream_audio.encode(frame):
                        container.mux(packet)

                for packet in stream_audio.encode(None):
                    container.mux(packet)

            for vid in video_tensor:
                frame = av.VideoFrame.from_ndarray(vid, format='rgb24')
                for packet in stream_video.encode(frame):
                    container.mux(packet)

            for packet in stream_video.encode(None):
                container.mux(packet)

    def to_bytes(
        self: 'T',
        audio_tensor: Optional[AudioTensor] = None,
        video_frame_rate: int = 24,
        video_codec: str = 'h264',
        audio_frame_rate: int = 48000,
        audio_codec: str = 'aac',
        audio_format: str = 'fltp',
    ) -> 'VideoBytes':
        """
        Convert video tensor to [`VideoBytes`][docarray.typing.VideoBytes].

        :param audio_tensor: AudioTensor containing the video's soundtrack.
        :param video_frame_rate: video frames per second.
        :param video_codec: the name of a video decoder/encoder.
        :param audio_frame_rate: audio frames per second.
        :param audio_codec: the name of an audio decoder/encoder.
        :param audio_format: the name of one of the audio formats supported by PyAV,
            such as 'flt', 'fltp', 's16' or 's16p'.

        :return: a VideoBytes object
        """
        from docarray.typing.bytes.video_bytes import VideoBytes

        bytes = BytesIO()
        self.save(
            file_path=bytes,
            audio_tensor=audio_tensor,
            video_frame_rate=video_frame_rate,
            video_codec=video_codec,
            audio_frame_rate=audio_frame_rate,
            audio_codec=audio_codec,
            audio_format=audio_format,
        )
        return VideoBytes(bytes.getvalue())

    def display(self, audio: Optional[AudioTensor] = None) -> None:
        """
        Display video data from tensor in notebook.

        :param audio: sound to play with video tensor
        """
        if is_notebook():
            from IPython.display import Video, display

            b = self.to_bytes(audio_tensor=audio)
            display(Video(data=b, embed=True, mimetype='video/mp4'))
        else:
            warnings.warn('Display of video is only possible in a notebook.')
