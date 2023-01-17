from typing import TYPE_CHECKING, BinaryIO, Optional, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.audio.audio_tensor import AudioTensor

if TYPE_CHECKING:
    from docarray.typing import VideoNdArray, VideoTorchTensor


T = TypeVar('T', bound='VideoTensorMixin')


class VideoTensorMixin:
    @staticmethod
    def validate_shape(
        cls: Union[Type['VideoTorchTensor'], Type['VideoNdArray']], value: 'T'
    ) -> 'T':
        comp_backend = cls.get_comp_backend()
        shape = comp_backend.shape(value)  # type: ignore
        if comp_backend.n_dim(value) not in [3, 4] or shape[-1] != 3:  # type: ignore
            raise ValueError(
                f'Expects tensor with 3 or 4 dimensions and the last dimension equal '
                f'to 3, but received {shape}.'
            )
        else:
            return value

    def save(
        self: 'T',
        file_path: Union[str, BinaryIO],
        audio_tensor: Optional[AudioTensor] = None,
        video_frame_rate: int = 24,
        video_codec: str = 'h264',
        audio_frame_rate: int = 48000,
        audio_codec: str = 'aac',
        audio_format: str = 'fltp',
    ) -> None:
        """
        Save video tensor to a .mp4 file.

        :param file_path: path to a .mp4 file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param audio_tensor: AudioTensor containing the video's soundtrack.
        :param video_frame_rate: video frames per second.
        :param video_codec: the name of a video decoder/encoder.
        :param audio_frame_rate: audio frames per second.
        :param audio_codec: the name of an audio decoder/encoder.
        :param audio_format: the name of one of the audio formats supported by PyAV,
            such as 'flt', 'fltp', 's16' or 's16p'.

        EXAMPLE USAGE

        .. code-block:: python
            import numpy as np

            from docarray import BaseDocument
            from docarray.typing.tensor.audio.audio_tensor import AudioTensor
            from docarray.typing.tensor.video.video_tensor import VideoTensor


            class MyDoc(BaseDocument):
                video_tensor: VideoTensor
                audio_tensor: AudioTensor


            doc = MyDoc(
                video_tensor=np.random.randint(low=0, high=256, size=(10, 200, 300, 3)),
                audio_tensor=np.random.randn(100, 1, 1024).astype("float32"),
            )

            doc.video_tensor.save(
                file_path="toydata/mp_.mp4",
                audio_tensor=doc.audio_tensor,
                audio_format="flt",
            )

        """
        import av

        np_tensor = self.get_comp_backend().to_numpy(array=self)  # type: ignore
        video_tensor = np_tensor.astype('uint8')

        with av.open(file_path, mode='w') as container:
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
