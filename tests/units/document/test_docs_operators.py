from docarray.documents.text import Text


def test_text_document_operators():
    doc = Text(text='text', url='url.com')

    assert doc == 'text'
    assert doc != 'url.com'

    doc2 = Text(id=doc.id, text='text', url='url.com')
    assert doc == doc2

    doc3 = Text(id='other-id', text='text', url='url.com')
    assert doc != doc3

    assert 't' in doc
    assert 'a' not in doc

    t = Text(text='this is my text document')
    assert 'text' in t
    assert 'docarray' not in t

    text = Text()
    assert text is not None
    assert text.text is None
