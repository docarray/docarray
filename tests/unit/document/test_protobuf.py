import pytest

from docarray import Document
from docarray.score import NamedScore


@pytest.mark.parametrize('doc', [
    Document(tags={'hello': 'world', 'sad': {'nest': 123}, 'hello12': 1.2}),
    Document(scores={'hello': NamedScore(value=1., description='hello')}),
    Document(location=[1., 2., 3.])
])
def test_to_from_protobuf(doc):
    docr = Document.from_protobuf(doc.to_protobuf())
    assert docr == doc
