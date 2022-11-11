from docarray.document.document import BaseDocument


def test_base_document_init():

    doc = BaseDocument()

    assert doc.id is not None
