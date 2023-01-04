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
        frame_rate: int = 30,
        codec: str = 'h264',
    ) -> None:
        """
        Save video tensor to a .wav file. Mono/stereo is preserved.


        :param file_path: path to a .wav file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param frame_rate: frames per second.
        :param codec: the name of a decoder/encoder.
        """
        np_tensor = self.to_numpy()

        video_tensor = np.moveaxis(np.clip(np_tensor, 0, 255), 1, 2).astype('uint8')

        import av

        with av.open(file_path, mode='w') as container:
            stream = container.add_stream(codec, rate=frame_rate)
            stream.width = np_tensor.shape[1]
            stream.height = np_tensor.shape[2]
            stream.pix_fmt = 'yuv420p'

            for b in video_tensor:
                frame = av.VideoFrame.from_ndarray(b, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
