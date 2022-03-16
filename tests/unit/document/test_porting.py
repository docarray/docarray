import json

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


@pytest.mark.parametrize('to_fn,preproc', [('dict', dict), ('json', json.dumps)])
def test_schemaless(to_fn, preproc):
    input = {
        'attr1': 123,
        'attr2': 'abc',
        'attr3': [1, 2, 3],
        'attr4': ['a', 'b', 'c'],
        'attr5': {
            'attr6': 'a',
            'attr7': 1,
        },
    }
    doc = getattr(Document, f'from_{to_fn}')(preproc(input), protocol=None)
    assert doc.tags['attr1'] == 123
    assert doc.tags['attr2'] == 'abc'
    assert doc.tags['attr3'] == [1, 2, 3]
    assert doc.tags['attr4'] == ['a', 'b', 'c']

    assert doc.tags['attr5'] == {
        'attr6': 'a',
        'attr7': 1,
    }


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_base64(protocol, compress):
    d = Document(text='hello', embedding=[1, 2, 3])
    d_r = Document.from_base64(d.to_base64(protocol, compress), protocol, compress)
    assert d_r == d
    assert d_r.embedding == [1, 2, 3]
