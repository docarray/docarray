import wave
from abc import ABC, abstractmethod
from typing import BinaryIO, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='AbstractAudioTensor')


class AbstractAudioTensor(AbstractTensor, ABC):
    @abstractmethod
    def to_bytes(self):
        """
        Convert audio tensor to bytes.
        """
        ...

    def save_to_wav_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        sample_rate: int = 44100,
        sample_width: int = 2,
    ) -> None:
        """
        Save audio tensor to a .wav file. Mono/stereo is preserved.

        :param file_path: path to a .wav file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param sample_rate: sampling frequency
        :param sample_width: sample width in bytes
        """
        comp_backend = self.get_comp_backend()
        n_channels = 2 if comp_backend.n_dim(array=self) > 1 else 1  # type: ignore

        with wave.open(file_path, 'w') as f:
            f.setnchannels(n_channels)
            f.setsampwidth(sample_width)
            f.setframerate(sample_rate)
            f.writeframes(self.to_bytes())
