from abc import ABC
from typing import BinaryIO, Optional, TypeVar, Union

import numpy as np

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.audio.audio_tensor import AudioTensor

T = TypeVar('T', bound='AbstractVideoTensor')


class AbstractVideoTensor(AbstractTensor, ABC):
    def save_to_mp4_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        audio_tensor: Optional[AudioTensor] = None,
        video_frame_rate: int = 30,
        video_codec: str = 'h264',
        audio_frame_rate: int = 48000,
        audio_codec: str = 'aac',
        audio_format: str = 'fltp',
    ) -> None:
        """
        Save video tensor to a .mp4 file.

        :param file_path: path to a .mp4 file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param video_frame_rate: frames per second.
        :param video_codec: the name of a decoder/encoder.
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
                    for packet in stream_audio.encode(frame):
                        container.mux(packet)

            for vid in video_tensor:
                frame = av.VideoFrame.from_ndarray(vid, format='rgb24')
                for packet in stream_video.encode(frame):
                    container.mux(packet)

            for packet in stream_audio.encode(None):
                container.mux(packet)
            for packet in stream_video.encode(None):
                container.mux(packet)
