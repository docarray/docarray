from docarray.documents import TextDoc


def test_text_document_init():
    text = TextDoc('hello world')
    assert text.text == 'hello world'
    assert text == 'hello world'

    text = TextDoc(text='hello world')
    assert text.text == 'hello world'
    assert text == 'hello world'

    text = TextDoc()
    assert text is not None
    assert text.text is None
