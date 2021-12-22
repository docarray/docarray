import numpy as np
import pytest

from docarray import Document
from docarray.score import NamedScore


@pytest.mark.parametrize(
    'doc',
    [
        Document(tags={'hello': 'world', 'sad': {'nest': 123}, 'hello12': 1.2}),
        Document(scores={'hello': NamedScore(value=1.0, description='hello')}),
        Document(location=[1.0, 2.0, 3.0]),
        Document(chunks=[Document()], matches=[Document(), Document()]),
    ],
)
def test_to_from_protobuf(doc):
    docr = Document.from_protobuf(doc.to_protobuf())
    assert docr == doc


def test_to_protobuf():
    with pytest.raises(TypeError):
        Document(text='hello', embedding=np.array([1, 2, 3]), id=1).to_protobuf()

    with pytest.raises(AttributeError):
        Document(tags=1).to_protobuf()

    assert (
        Document(text='hello', embedding=np.array([1, 2, 3])).to_protobuf().text
        == 'hello'
    )
    assert Document(tags={'hello': 'world'}).to_protobuf().tags
    assert len(Document(chunks=[Document(), Document()]).to_protobuf().chunks) == 2
