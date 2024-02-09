// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
from typing import Optional

import numpy as np
import pytest

from docarray import BaseDoc, DocList, DocVec
from docarray.typing import NdArray


class Nested(BaseDoc):
    tensor: NdArray


class Image(BaseDoc):
    features: Optional[Nested] = None


def test_optional_field():
    docs = DocVec[Image]([Image() for _ in range(10)])

    assert docs.features is None

    docs.features = DocList[Nested]([Nested(tensor=np.zeros(10)) for _ in range(10)])

    assert docs.features.tensor.shape == (10, 10)

    for doc in docs:
        assert doc.features.tensor.shape == (10,)


def test_set_none():
    docs = DocVec[Image](
        [Image(features=Nested(tensor=np.zeros(10))) for _ in range(10)]
    )
    assert docs.features.tensor.shape == (10, 10)

    docs.features = None

    assert docs.features is None

    for doc in docs:
        assert doc.features is None


def test_set_doc():
    docs = DocVec[Image](
        [Image(features=Nested(tensor=np.zeros(10))) for _ in range(10)]
    )
    assert docs.features.tensor.shape == (10, 10)

    for doc in docs:
        doc.features = Nested(tensor=np.ones(10))

        with pytest.raises(ValueError):
            doc.features = None


def test_set_doc_none():
    docs = DocVec[Image]([Image() for _ in range(10)])

    assert docs.features is None

    for doc in docs:
        with pytest.raises(ValueError):
            doc.features = Nested(tensor=np.ones(10))


def test_no_uniform_none():
    with pytest.raises(ValueError):
        DocVec[Image]([Image(), Image(features=Nested(tensor=np.zeros(10)))])

    with pytest.raises(ValueError):
        DocVec[Image]([Image(features=Nested(tensor=np.zeros(10))), Image()])
