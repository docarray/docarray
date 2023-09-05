from typing import Dict, List

import numpy as np
import pytest
from orjson import orjson

from docarray import DocList
from docarray.base_doc import AnyDoc, BaseDoc
from docarray.base_doc.io.json import orjson_dumps_and_decode
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.pydantic import is_pydantic_v2


def test_any_doc():
    class InnerDocument(BaseDoc):
        text: str
        tensor: NdArray

    class CustomDoc(BaseDoc):
        inner: InnerDocument
        text: str

    doc = CustomDoc(
        text='bye', inner=InnerDocument(text='hello', tensor=np.zeros((3, 224, 224)))
    )

    any_doc = AnyDoc(**doc.__dict__)

    assert any_doc.text == doc.text
    assert any_doc.inner.text == doc.inner.text
    assert (any_doc.inner.tensor == doc.inner.tensor).all()


@pytest.mark.parametrize('protocol', ['proto', 'json'])
def test_any_document_from_to(protocol):
    class InnerDoc(BaseDoc):
        text: str
        t: Dict[str, str]

    class DocTest(BaseDoc):
        text: str
        tags: Dict[str, int]
        l_: List[int]
        d: InnerDoc
        ld: DocList[InnerDoc]

    inner_doc = InnerDoc(text='I am inner', t={'a': 'b'})
    da = DocList[DocTest](
        [
            DocTest(
                text='type1',
                tags={'type': 1},
                l_=[1, 2],
                d=inner_doc,
                ld=DocList[InnerDoc]([inner_doc]),
            ),
            DocTest(
                text='type2',
                tags={'type': 2},
                l_=[1, 2],
                d=inner_doc,
                ld=DocList[InnerDoc]([inner_doc]),
            ),
        ]
    )

    from docarray.base_doc import AnyDoc

    if protocol == 'proto':
        aux = DocList[AnyDoc].from_protobuf(da.to_protobuf())
    else:
        aux = DocList[AnyDoc].from_json(da.to_json())
    assert len(aux) == 2
    assert len(aux.id) == 2
    for i, d in enumerate(aux):
        assert d.tags['type'] == i + 1
        assert d.text == f'type{i + 1}'
        assert d.l_ == [1, 2]
        if protocol == 'proto':
            assert isinstance(d.d, AnyDoc)
            assert d.d.text == 'I am inner'  # inner Document is a Dict
            assert d.d.t == {'a': 'b'}
        else:
            assert isinstance(d.d, dict)
            assert d.d['text'] == 'I am inner'  # inner Document is a Dict
            assert d.d['t'] == {'a': 'b'}
        assert len(d.ld) == 1
        if protocol == 'proto':
            assert isinstance(d.ld[0], AnyDoc)
            assert d.ld[0].text == 'I am inner'
            assert d.ld[0].t == {'a': 'b'}
        else:
            assert isinstance(d.ld[0], dict)
            assert d.ld[0]['text'] == 'I am inner'
            assert d.ld[0]['t'] == {'a': 'b'}


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_subclass_config():
    class MyDoc(BaseDoc):
        x: str

        class Config(BaseDoc.Config):
            arbitrary_types_allowed = True  # just an example setting

    assert MyDoc.Config.json_loads == orjson.loads
    assert MyDoc.Config.json_dumps == orjson_dumps_and_decode
    assert (
        MyDoc.Config.json_encoders[AbstractTensor](3) == 3
    )  # dirty check that it is identity
    assert MyDoc.Config.validate_assignment
    assert not MyDoc.Config._load_extra_fields_from_protobuf
    assert MyDoc.Config.arbitrary_types_allowed
