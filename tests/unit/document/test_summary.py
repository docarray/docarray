import os

import numpy as np
import pytest

from docarray import Document

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_single_doc_summary():
    # empty doc
    Document().summary()
    # nested doc
    Document(
        chunks=[
            Document(),
            Document(chunks=[Document()]),
            Document(),
        ],
        matches=[Document(), Document()],
    ).summary()


def test_plot_image():
    d = Document(uri=os.path.join(cur_dir, 'toydata/test.png'))
    d.display()
    d.display(from_='uri')

    d.load_uri_to_image_tensor()
    d.uri = None

    d.display()
    d.display(from_='tensor')


def test_plot_image_wrong():
    d = Document(uri=os.path.join(cur_dir, 'toydata/test.png'))
    with pytest.raises(ValueError):
        d.display(from_='tensor')

    d.load_uri_to_image_tensor()
    d.uri = None

    with pytest.raises(ValueError):
        d.display(from_='uri')


def test_plot_audio():
    d = Document(uri=os.path.join(cur_dir, 'toydata/hello.wav'))
    d.display()

    d.convert_uri_to_datauri()
    d.display()


def test_plot_video():
    d = Document(uri=os.path.join(cur_dir, 'toydata/mov_bbb.mp4'))
    d.display()

    d.convert_uri_to_datauri()
    d.display()


def test_plot_embedding():
    d = Document(embedding=[1, 2, 3], tensor=np.random.random(128))
    d.summary()

    c = Document(embedding=[1, 2, 3], tensor=np.random.random(128))
    d.chunks.append(c)
    d.summary()
