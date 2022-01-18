import pytest

from docarray import Document, DocumentArray
from tests import random_docs


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = Document(embedding=[1, 2, 3, 4, 5], text='hello')
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    d2 = Document.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.non_empty_fields == d.non_empty_fields


@pytest.mark.parametrize('target', [DocumentArray.empty(10), random_docs(10)])
@pytest.mark.parametrize('protocol', ['jsonschema', 'protobuf'])
@pytest.mark.parametrize('to_fn', ['dict', 'json'])
def test_dict_json(target, protocol, to_fn):
    for d in target:
        d_r = getattr(Document, f'from_{to_fn}')(
            getattr(d, f'to_{to_fn}')(protocol=protocol), protocol=protocol
        )
        assert d == d_r


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_base64(protocol, compress):
    d = Document(text='hello', embedding=[1, 2, 3])
    d_r = Document.from_base64(d.to_base64(protocol, compress), protocol, compress)
    assert d_r == d
    assert d_r.embedding == [1, 2, 3]


@pytest.mark.parametrize('protocol', ['protobuf', 'jsonschema'])
def test_tags_type_mantained(protocol):
    d = Document(tags={'a': 100})
    assert isinstance(d.to_dict(protocol=protocol)['tags']['a'], float)
