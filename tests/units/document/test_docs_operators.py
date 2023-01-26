from docarray.documents.text import Text


def test_text_document_operators():

    doc = Text(text='text', url='url.com')

    assert doc == 'text'
    assert doc != 'url.com'

    doc2 = Text(text='text', url='url.com')
    assert doc == doc2
