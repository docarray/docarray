import numpy as np
import pytest

from docarray.documents import Image

REMOTE_JPG = (
    'https://upload.wikimedia.org/wikipedia/commons/8/80/'
    'Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg'
)


@pytest.mark.slow
@pytest.mark.internet
def test_image():

    image = Image(url=REMOTE_JPG)

    image.tensor = image.url.load()

    assert isinstance(image.tensor, np.ndarray)
