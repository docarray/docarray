import numpy as np
import pytest

from docarray import DocumentArray
from docarray.documents import ImageDoc

REMOTE_JPG = (
    'https://upload.wikimedia.org/wikipedia/commons/8/80/'
    'Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg'
)


@pytest.mark.slow
@pytest.mark.internet
def test_image():

    da = DocumentArray[ImageDoc]([ImageDoc(url=REMOTE_JPG) for _ in range(3)])

    da.tensor = da.url.load()

    for doc in da:
        assert isinstance(doc.tensor, np.ndarray)
