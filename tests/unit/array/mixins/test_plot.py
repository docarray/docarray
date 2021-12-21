import json
import os
import random

import numpy as np
import pytest

from docarray import DocumentArray, Document


def test_sprite_image_generator(pytestconfig, tmpdir):
    da = DocumentArray.from_files(
        [
            f'{pytestconfig.rootdir}/**/*.png',
            f'{pytestconfig.rootdir}/**/*.jpg',
            f'{pytestconfig.rootdir}/**/*.jpeg',
        ]
    )
    da.plot_image_sprites(tmpdir / 'sprint_da.png')
    assert os.path.exists(tmpdir / 'sprint_da.png')


def da_and_dam():
    embeddings = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    doc_array = DocumentArray(
        [
            Document(embedding=x, tags={'label': random.randint(0, 5)})
            for x in embeddings
        ]
    )

    return (doc_array,)


@pytest.mark.parametrize('da', da_and_dam())
def test_plot_embeddings(da):
    p = da.plot_embeddings(start_server=False)
    assert os.path.exists(p)
    assert os.path.exists(os.path.join(p, 'config.json'))
    with open(os.path.join(p, 'config.json')) as fp:
        config = json.load(fp)
        assert len(config['embeddings']) == 1
        assert config['embeddings'][0]['tensorShape'] == list(da.embeddings.shape)


def test_plot_embeddings_same_path(tmpdir):
    da1 = DocumentArray.empty(100)
    da1.embeddings = np.random.random([100, 5])
    p1 = da1.plot_embeddings(start_server=False, path=tmpdir)
    da2 = DocumentArray.empty(768)
    da2.embeddings = np.random.random([768, 5])
    p2 = da2.plot_embeddings(start_server=False, path=tmpdir)
    assert p1 == p2
    assert os.path.exists(p1)
    with open(os.path.join(p1, 'config.json')) as fp:
        config = json.load(fp)
        assert len(config['embeddings']) == 2
