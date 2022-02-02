import numpy as np
import pytest

from docarray import DocumentArray, Document


@pytest.fixture
def docs():
    yield (Document(text=j) for j in range(100))


@pytest.fixture
def indices():
    yield (i for i in [-2, 0, 2])
