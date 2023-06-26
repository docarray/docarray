from docarray.documents.text import TextDoc


def test_text_document_operators():
    doc = TextDoc(text='text', url='http://url.com/file.txt')

    assert doc == 'text'
    assert doc != 'http://url.com'

    doc2 = TextDoc(id=doc.id, text='text', url='http://url.com/file.txt')
    assert doc == doc2

    doc3 = TextDoc(id='other-id', text='text', url='http://url.com/file.txt')
    assert doc == doc3

    assert 't' in doc
    assert 'a' not in doc

    t = TextDoc(text='this is my text document')
    assert 'text' in t
    assert 'docarray' not in t

    text = TextDoc()
    assert text is not None
    assert text.text is None
