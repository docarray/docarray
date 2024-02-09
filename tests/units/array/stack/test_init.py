# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from docarray import BaseDoc
from docarray.array.doc_vec.doc_vec import DocVec
from docarray.typing import AnyTensor, NdArray


def test_da_init():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros(10), name='hello') for _ in range(4)]

    da = DocVec[MyDoc](docs, tensor_type=NdArray)

    assert (da._storage.tensor_columns['tensor'] == np.zeros((4, 10))).all()
    assert da._storage.any_columns['name'] == ['hello' for _ in range(4)]


def test_da_iter():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=i * np.zeros((10, 10)), name=f'hello{i}') for i in range(4)]

    da = DocVec[MyDoc](docs, tensor_type=NdArray)

    for i, doc in enumerate(da):
        assert isinstance(doc, MyDoc)
        assert (doc.tensor == i * np.zeros((10, 10))).all()
        assert doc.name == f'hello{i}'
