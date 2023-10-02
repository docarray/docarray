from typing import Optional

import numpy as np

from docarray import BaseDoc, DocVec
from docarray.typing import ImageUrl, NdArray


def test_optional():
    class Features(BaseDoc):
        tensor: NdArray[100]

    class Image(BaseDoc):
        url: ImageUrl
        features: Optional[Features] = None

    docs = DocVec[Image]([Image(url='http://url.com/foo.png') for _ in range(10)])

    print(docs.features)  # None

    docs.features = [Features(tensor=np.random.random([100])) for _ in range(10)]
    print(docs.features)  # <DocVec[Features] (length=10)>
    assert isinstance(docs.features, DocVec[Features])

    docs.features.tensor = np.ones((10, 100))

    assert docs[0].features.tensor.shape == (100,)

    docs.features = None

    assert docs[0].features is None
