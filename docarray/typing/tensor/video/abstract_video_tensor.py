from abc import ABC, abstractmethod
from typing import BinaryIO, TypeVar, Union

import numpy as np

from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='AbstractVideoTensor')


class AbstractVideoTensor(AbstractTensor, ABC):
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """
        Convert video tensor to numpy.ndarray.
        """
        ...

    def save_to_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        frame_rate: int = 24,
        codec: str = 'h264',
    ) -> None:
        """
        Save video tensor to a .mp4 file.

        :param file_path: path to a .mp4 file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param frame_rate: frames per second.
        :param codec: the name of a decoder/encoder.
        """
        np_tensor = self.to_numpy()
        print(f"np_tensor[0][:2] = {np_tensor[0][:2]}")
        video_tensor = np_tensor.astype('uint8')
        import av

        with av.open(file_path, mode='w') as container:
            if video_tensor.ndim == 3:
                video_tensor = np.expand_dims(video_tensor, axis=0)

            stream = container.add_stream(codec, rate=frame_rate)
            stream.height = video_tensor.shape[-3]
            stream.width = video_tensor.shape[-2]

            for vid in video_tensor:
                frame = av.VideoFrame.from_ndarray(vid, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)
