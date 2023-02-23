import warnings
import wave
from abc import ABC
from typing import BinaryIO, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.misc import is_notebook

T = TypeVar('T', bound='AbstractAudioTensor')

MAX_INT_16 = 2**15


class AbstractAudioTensor(AbstractTensor, ABC):
    def to_bytes(self):
        """
        Convert audio tensor to bytes.
        """
        tensor = self.get_comp_backend().to_numpy(self)
        tensor = (tensor * MAX_INT_16).astype('<h')
        return tensor.tobytes()

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

    def save(
        self: 'T',
        file_path: Union[str, BinaryIO],
        format: str = 'wav',
        frame_rate: int = 44100,
        sample_width: int = 2,
        *args,
        **kwargs
    ) -> None:
        """
        Save audio tensor to an audio file. Mono/stereo is preserved.

        :param file_path: path to an audio file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param format: format for the audio file ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)
        :param frame_rate: sampling frequency
        :param sample_width: sample width in bytes
        """
        from pydub import AudioSegment  # type: ignore

        comp_backend = self.get_comp_backend()
        channels = 2 if comp_backend.n_dim(array=self) > 1 else 1  # type: ignore

        segment = AudioSegment(
            self.to_bytes(),
            frame_rate=frame_rate,
            sample_width=sample_width,
            channels=channels,
        )
        segment.export(file_path, format=format, *args, **kwargs)

    def display(self, rate=44100):
        """
        Play audio data from tensor in notebook.
        """
        if is_notebook():
            from IPython.display import Audio, display

            audio_np = self.get_comp_backend().to_numpy(self)
            display(Audio(audio_np, rate=rate))
        else:
            warnings.warn('Display of audio is only possible in a notebook.')
