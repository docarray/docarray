from collections import defaultdict

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


@pytest.mark.parametrize('meth', ['protobuf', 'dict'])
@pytest.mark.parametrize('attr', ['scores', 'evaluations'])
def test_from_to_namescore_default_dict(attr, meth):
    d = Document()
    getattr(d, attr)['relevance'].value = 3.0
    assert isinstance(d.scores, defaultdict)

    r_d = getattr(Document, f'from_{meth}')(getattr(d, f'to_{meth}')())
    assert isinstance(r_d.scores, defaultdict)
