import os

from docarray import Document

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_single_doc_summary():
    # empty doc
    Document().summary()
    # nested doc
    Document(
        chunks=[
            Document(),
            Document(chunks=[Document()]),
            Document(),
        ],
        matches=[Document(), Document()],
    ).summary()


def test_plot_image():
    d = Document(uri=os.path.join(cur_dir, 'toydata/test.png'))
    d.plot_image()

    d.load_uri_to_image_blob()
    d.uri = None

    d.plot_image()
