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
import numpy as np

from docarray import BaseDoc
from docarray.array import DocVec
from docarray.array.doc_vec.column_storage import ColumnStorageView
from docarray.typing import AnyTensor


def test_document_view():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=str(i)) for i in range(4)]

    doc_vec = DocVec[MyDoc](docs)
    storage = doc_vec._storage

    result = str(doc_vec[0])
    assert 'MyDoc' in result
    assert 'id' in result
    assert 'tensor' in result
    assert 'name' in result

    doc = MyDoc.from_view(ColumnStorageView(0, storage))
    assert doc.is_view()
    assert doc.id == '0'
    assert (doc.tensor == np.zeros(10)).all()
    assert doc.name == 'hello'

    storage.columns['id'][0] = '12345'
    storage.columns['tensor'][0] = np.ones(10)
    storage.columns['name'][0] = 'byebye'

    assert doc.id == '12345'
    assert (doc.tensor == np.ones(10)).all()
    assert doc.name == 'byebye'
