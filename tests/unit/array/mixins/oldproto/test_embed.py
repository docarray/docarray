import numpy as np
import pytest

from docarray import DocumentArray


@pytest.fixture
def paddle_model():
    import paddle.nn as nn

    class DummyPaddleLayer(nn.Layer):
        def forward(self, x, y):
            return (x + y) / 2.0

    return DummyPaddleLayer()


def test_embeded_paddle_model(paddle_model):
    def collate_fn(da):
        return {'x': da.tensors, 'y': da.tensors}

    docs = DocumentArray.empty(3)
    docs.tensors = np.random.random([3, 5]).astype(np.float32)
    docs.embed(paddle_model, collate_fn=collate_fn, to_numpy=True)
    assert (docs.tensors == docs.embeddings).all()
