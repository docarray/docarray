import pytest
from docarray import BaseDoc, DocList
from docarray.utils._internal.pydantic import is_pydantic_v2


@pytest.mark.skipif(not is_pydantic_v2, reason='Feature only available for Pydantic V2')
def test_schema_nested():
    # check issue https://github.com/docarray/docarray/issues/1521

    class Doc1Test(BaseDoc):
        aux: str

    class DocDocTest(BaseDoc):
        docs: DocList[Doc1Test]

    assert 'Doc1Test' in DocDocTest.schema()['$defs']
    d = DocDocTest(docs=DocList[Doc1Test]([Doc1Test(aux='aux')]))

    assert type(d.docs) == DocList[Doc1Test]
    assert d.docs.aux == ['aux']
