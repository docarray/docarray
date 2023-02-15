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

    def display(self, rate=44100):
        audio_np = self.get_comp_backend().to_numpy(self)
        if is_notebook():
            from IPython.display import Audio, display

            display(Audio(audio_np, rate=rate))
        else:
            # b = self.load()
            # res = requests.get(self)
            # print(f"type(res.text) = {type(res.text)}")
            # print(f"type(res.content) = {type(res.content)}")
            # sound = AudioSegment.from_file(BytesIO(res.content), "wav")
            # sound = AudioSegment.from_file(self, format="wav")
            import os
            import tempfile

            # sound = AudioSegment.from_file(self, format="wav")
            # raise NotImplementedError
            # from io import BytesIO
            # import requests
            from pydub import AudioSegment
            from pydub.playback import play

            tmp = tempfile.NamedTemporaryFile(delete=False)
            try:
                self.save_to_wav_file(tmp)
                sound = AudioSegment.from_file(tmp, "wav")
                play(sound)
            finally:
                tmp.close()
                os.unlink(tmp.name)
