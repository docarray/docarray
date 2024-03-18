from typing import Any, Dict, List, Optional, Union, ClassVar

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from docarray.typing import AnyTensor, ImageUrl
from docarray.utils.create_dynamic_doc_class import (
    create_base_doc_from_schema,
    create_pure_python_type_model,
)
from docarray.utils._internal.pydantic import is_pydantic_v2


@pytest.mark.parametrize('transformation', ['proto', 'json'])
def test_create_pydantic_model_from_schema(transformation):
    class Nested2Doc(BaseDoc):
        value: str
        classvar: ClassVar[str] = 'classvar2'

    class Nested1Doc(BaseDoc):
        nested: Nested2Doc
        classvar: ClassVar[str] = 'classvar1'

    class CustomDoc(BaseDoc):
        tensor: Optional[AnyTensor] = None
        url: ImageUrl
        num: float = 0.5
        num_num: List[float] = [1.5, 2.5]
        lll: List[List[List[int]]] = [[[5]]]
        fff: List[List[List[float]]] = [[[5.2]]]
        single_text: TextDoc
        texts: DocList[TextDoc]
        d: Dict[str, str] = {'a': 'b'}
        di: Optional[Dict[str, int]] = None
        u: Union[str, int]
        lu: List[Union[str, int]] = [0, 1, 2]
        tags: Optional[Dict[str, Any]] = None
        nested: Nested1Doc
        classvar: ClassVar[str] = 'classvar'

    CustomDocCopy = create_pure_python_type_model(CustomDoc)
    new_custom_doc_model = create_base_doc_from_schema(
        CustomDocCopy.schema(), 'CustomDoc', {}
    )

    original_custom_docs = DocList[CustomDoc](
        [
            CustomDoc(
                num=3.5,
                num_num=[4.5, 5.5],
                url='photo.jpg',
                lll=[[[40]]],
                fff=[[[40.2]]],
                d={'b': 'a'},
                texts=DocList[TextDoc]([TextDoc(text='hey ha', embedding=np.zeros(3))]),
                single_text=TextDoc(text='single hey ha', embedding=np.zeros(2)),
                u='a',
                lu=[3, 4],
                nested=Nested1Doc(nested=Nested2Doc(value='hello world')),
            )
        ]
    )
    for doc in original_custom_docs:
        doc.tensor = np.zeros((10, 10, 10))
        doc.di = {'a': 2}

    if transformation == 'proto':
        custom_partial_da = DocList[new_custom_doc_model].from_protobuf(
            original_custom_docs.to_protobuf()
        )
        original_back = DocList[CustomDoc].from_protobuf(
            custom_partial_da.to_protobuf()
        )
    elif transformation == 'json':
        custom_partial_da = DocList[new_custom_doc_model].from_json(
            original_custom_docs.to_json()
        )
        original_back = DocList[CustomDoc].from_json(custom_partial_da.to_json())

    assert len(custom_partial_da) == 1
    assert custom_partial_da[0].url == 'photo.jpg'
    assert custom_partial_da[0].num == 3.5
    assert custom_partial_da[0].num_num == [4.5, 5.5]
    assert custom_partial_da[0].lll == [[[40]]]
    if is_pydantic_v2:
        assert custom_partial_da[0].lu == [3, 4]
    else:
        assert custom_partial_da[0].lu == ['3', '4']  # Union validates back to string
    assert custom_partial_da[0].fff == [[[40.2]]]
    assert custom_partial_da[0].di == {'a': 2}
    assert custom_partial_da[0].d == {'b': 'a'}
    assert len(custom_partial_da[0].texts) == 1
    assert custom_partial_da[0].texts[0].text == 'hey ha'
    assert custom_partial_da[0].texts[0].embedding.shape == (3,)
    assert custom_partial_da[0].tensor.shape == (10, 10, 10)
    assert custom_partial_da[0].u == 'a'
    assert custom_partial_da[0].single_text.text == 'single hey ha'
    assert custom_partial_da[0].single_text.embedding.shape == (2,)
    assert original_back[0].nested.nested.value == 'hello world'
    assert original_back[0].num == 3.5
    assert original_back[0].num_num == [4.5, 5.5]
    assert original_back[0].classvar == 'classvar'
    assert original_back[0].nested.classvar == 'classvar1'
    assert original_back[0].nested.nested.classvar == 'classvar2'

    assert len(original_back) == 1
    assert original_back[0].url == 'photo.jpg'
    assert original_back[0].lll == [[[40]]]
    if is_pydantic_v2:
        assert original_back[0].lu == [3, 4]  # Union validates back to string
    else:
        assert original_back[0].lu == ['3', '4']  # Union validates back to string
    assert original_back[0].fff == [[[40.2]]]
    assert original_back[0].di == {'a': 2}
    assert original_back[0].d == {'b': 'a'}
    assert len(original_back[0].texts) == 1
    assert original_back[0].texts[0].text == 'hey ha'
    assert original_back[0].texts[0].embedding.shape == (3,)
    assert original_back[0].tensor.shape == (10, 10, 10)
    assert original_back[0].u == 'a'
    assert original_back[0].single_text.text == 'single hey ha'
    assert original_back[0].single_text.embedding.shape == (2,)

    class TextDocWithId(BaseDoc):
        ia: str

    TextDocWithIdCopy = create_pure_python_type_model(TextDocWithId)
    new_textdoc_with_id_model = create_base_doc_from_schema(
        TextDocWithIdCopy.schema(), 'TextDocWithId', {}
    )

    original_text_doc_with_id = DocList[TextDocWithId](
        [TextDocWithId(ia=f'ID {i}') for i in range(10)]
    )
    if transformation == 'proto':
        custom_da = DocList[new_textdoc_with_id_model].from_protobuf(
            original_text_doc_with_id.to_protobuf()
        )
        original_back = DocList[TextDocWithId].from_protobuf(custom_da.to_protobuf())
    elif transformation == 'json':
        custom_da = DocList[new_textdoc_with_id_model].from_json(
            original_text_doc_with_id.to_json()
        )
        original_back = DocList[TextDocWithId].from_json(custom_da.to_json())

    assert len(custom_da) == 10
    for i, doc in enumerate(custom_da):
        assert doc.ia == f'ID {i}'

    assert len(original_back) == 10
    for i, doc in enumerate(original_back):
        assert doc.ia == f'ID {i}'

    class ResultTestDoc(BaseDoc):
        matches: DocList[TextDocWithId]

    ResultTestDocCopy = create_pure_python_type_model(ResultTestDoc)
    new_result_test_doc_with_id_model = create_base_doc_from_schema(
        ResultTestDocCopy.schema(), 'ResultTestDoc', {}
    )
    result_test_docs = DocList[ResultTestDoc](
        [ResultTestDoc(matches=original_text_doc_with_id)]
    )

    if transformation == 'proto':
        custom_da = DocList[new_result_test_doc_with_id_model].from_protobuf(
            result_test_docs.to_protobuf()
        )
        original_back = DocList[ResultTestDoc].from_protobuf(custom_da.to_protobuf())
    elif transformation == 'json':
        custom_da = DocList[new_result_test_doc_with_id_model].from_json(
            result_test_docs.to_json()
        )
        original_back = DocList[ResultTestDoc].from_json(custom_da.to_json())

    assert len(custom_da) == 1
    assert len(custom_da[0].matches) == 10
    for i, doc in enumerate(custom_da[0].matches):
        assert doc.ia == f'ID {i}'

    assert len(original_back) == 1
    assert len(original_back[0].matches) == 10
    for i, doc in enumerate(original_back[0].matches):
        assert doc.ia == f'ID {i}'


@pytest.mark.parametrize('transformation', ['proto', 'json'])
def test_create_empty_doc_list_from_schema(transformation):
    class CustomDoc(BaseDoc):
        tensor: Optional[AnyTensor]
        url: ImageUrl
        lll: List[List[List[int]]] = [[[5]]]
        fff: List[List[List[float]]] = [[[5.2]]]
        single_text: TextDoc
        texts: DocList[TextDoc]
        d: Dict[str, str] = {'a': 'b'}
        di: Optional[Dict[str, int]] = None
        u: Union[str, int]
        lu: List[Union[str, int]] = [0, 1, 2]
        tags: Optional[Dict[str, Any]] = None
        lf: List[float] = [3.0, 4.1]

    CustomDocCopy = create_pure_python_type_model(CustomDoc)
    new_custom_doc_model = create_base_doc_from_schema(
        CustomDocCopy.schema(), 'CustomDoc'
    )

    original_custom_docs = DocList[CustomDoc]()
    if transformation == 'proto':
        custom_partial_da = DocList[new_custom_doc_model].from_protobuf(
            original_custom_docs.to_protobuf()
        )
        original_back = DocList[CustomDoc].from_protobuf(
            custom_partial_da.to_protobuf()
        )
    elif transformation == 'json':
        custom_partial_da = DocList[new_custom_doc_model].from_json(
            original_custom_docs.to_json()
        )
        original_back = DocList[CustomDoc].from_json(custom_partial_da.to_json())

    assert len(custom_partial_da) == 0
    assert len(original_back) == 0

    class TextDocWithId(BaseDoc):
        ia: str

    TextDocWithIdCopy = create_pure_python_type_model(TextDocWithId)
    new_textdoc_with_id_model = create_base_doc_from_schema(
        TextDocWithIdCopy.schema(), 'TextDocWithId', {}
    )

    original_text_doc_with_id = DocList[TextDocWithId]()
    if transformation == 'proto':
        custom_da = DocList[new_textdoc_with_id_model].from_protobuf(
            original_text_doc_with_id.to_protobuf()
        )
        original_back = DocList[TextDocWithId].from_protobuf(custom_da.to_protobuf())
    elif transformation == 'json':
        custom_da = DocList[new_textdoc_with_id_model].from_json(
            original_text_doc_with_id.to_json()
        )
        original_back = DocList[TextDocWithId].from_json(custom_da.to_json())

    assert len(original_back) == 0
    assert len(custom_da) == 0

    class ResultTestDoc(BaseDoc):
        matches: DocList[TextDocWithId]

    ResultTestDocCopy = create_pure_python_type_model(ResultTestDoc)
    new_result_test_doc_with_id_model = create_base_doc_from_schema(
        ResultTestDocCopy.schema(), 'ResultTestDoc', {}
    )
    result_test_docs = DocList[ResultTestDoc]()

    if transformation == 'proto':
        custom_da = DocList[new_result_test_doc_with_id_model].from_protobuf(
            result_test_docs.to_protobuf()
        )
        original_back = DocList[ResultTestDoc].from_protobuf(custom_da.to_protobuf())
    elif transformation == 'json':
        custom_da = DocList[new_result_test_doc_with_id_model].from_json(
            result_test_docs.to_json()
        )
        original_back = DocList[ResultTestDoc].from_json(custom_da.to_json())

    assert len(original_back) == 0
    assert len(custom_da) == 0


def test_create_with_field_info():
    class CustomDoc(BaseDoc):
        """Here I have the description of the class"""

        a: str = Field(examples=['Example here'], another_extra='I am another extra')

    CustomDocCopy = create_pure_python_type_model(CustomDoc)
    new_custom_doc_model = create_base_doc_from_schema(
        CustomDocCopy.schema(), 'CustomDoc'
    )
    assert new_custom_doc_model.schema().get('properties')['a']['examples'] == [
        'Example here'
    ]
    assert (
        new_custom_doc_model.schema().get('properties')['a']['another_extra']
        == 'I am another extra'
    )
    assert (
        new_custom_doc_model.schema().get('description')
        == 'Here I have the description of the class'
    )


def test_dynamic_class_creation_multiple_doclist_nested():
    from docarray import BaseDoc, DocList

    class MyTextDoc(BaseDoc):
        text: str

    class QuoteFile(BaseDoc):
        texts: DocList[MyTextDoc]

    class SearchResult(BaseDoc):
        results: DocList[QuoteFile] = None

    models_created_by_name = {}
    SearchResult_aux = create_pure_python_type_model(SearchResult)
    _ = create_base_doc_from_schema(
        SearchResult_aux.schema(), 'SearchResult', models_created_by_name
    )
    QuoteFile_reconstructed_in_gateway_from_Search_results = models_created_by_name[
        'QuoteFile'
    ]
    textlist = DocList[models_created_by_name['MyTextDoc']](
        [models_created_by_name['MyTextDoc'](id='11', text='hey')]
    )

    reconstructed_in_gateway_from_Search_results = (
        QuoteFile_reconstructed_in_gateway_from_Search_results(id='0', texts=textlist)
    )
    assert reconstructed_in_gateway_from_Search_results.texts[0].text == 'hey'
