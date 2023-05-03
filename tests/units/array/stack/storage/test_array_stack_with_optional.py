from typing import Optional

import numpy as np
import pytest

from docarray import BaseDoc, DocList, DocVec
from docarray.typing import NdArray


class Nested(BaseDoc):
    tensor: NdArray


class Image(BaseDoc):
    features: Optional[Nested] = None


def test_optional_field():
    docs = DocVec[Image]([Image() for _ in range(10)])

    assert docs.features is None

    docs.features = DocList[Nested]([Nested(tensor=np.zeros(10)) for _ in range(10)])

    assert docs.features.tensor.shape == (10, 10)

    for doc in docs:
        assert doc.features.tensor.shape == (10,)


def test_set_none():
    docs = DocVec[Image](
        [Image(features=Nested(tensor=np.zeros(10))) for _ in range(10)]
    )
    assert docs.features.tensor.shape == (10, 10)

    docs.features = None

    assert docs.features is None

    for doc in docs:
        assert doc.features is None


def test_set_doc():
    docs = DocVec[Image](
        [Image(features=Nested(tensor=np.zeros(10))) for _ in range(10)]
    )
    assert docs.features.tensor.shape == (10, 10)

    for doc in docs:
        doc.features = Nested(tensor=np.ones(10))

        with pytest.raises(ValueError):
            doc.features = None


def test_set_doc_none():
    docs = DocVec[Image]([Image() for _ in range(10)])

    assert docs.features is None

    for doc in docs:
        with pytest.raises(ValueError):
            doc.features = Nested(tensor=np.ones(10))


def test_no_uniform_none():
    with pytest.raises(ValueError):
        DocVec[Image]([Image(), Image(features=Nested(tensor=np.zeros(10)))])

    with pytest.raises(ValueError):
        DocVec[Image]([Image(features=Nested(tensor=np.zeros(10))), Image()])
