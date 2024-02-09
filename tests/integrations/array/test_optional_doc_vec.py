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
