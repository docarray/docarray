from abc import ABC, abstractmethod
from typing import BinaryIO, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='AbstractAudioTensor')


class AbstractAudioTensor(AbstractTensor, ABC):
    @abstractmethod
    def save_audio_tensor_to_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        sample_rate: int = 44100,
        sample_width: int = 2,
    ) -> None:
        """
        Save :attr:`.tensor` into a .wav file. Mono/stereo is preserved.

        :param file_path: if file is a string, open the file by that name, otherwise
            treat it as a file-like object.
        :param sample_rate: sampling frequency
        :param sample_width: sample width in bytes
        """
        ...
