import copy
import os
import pytest

from docarray.document.generators import from_files

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_librosa_and_wave_compatibility(pytestconfig):
    for d in from_files(f'{cur_dir}/toydata/*.wav'):
        copy_d = copy.deepcopy(d)
        d._load_uri_to_audio_tensor_wave()
        copy_d._load_uri_to_audio_tensor_librosa()

        assert (d.tensor == copy_d.tensor).all()


@pytest.mark.parametrize('sr', [None, 16_000, 44_000])
def test_load_audio(pytestconfig, sr):
    for d in from_files(f'{cur_dir}/toydata/*.wav'):
        d.load_uri_to_audio_tensor(sample_rate=sr)
