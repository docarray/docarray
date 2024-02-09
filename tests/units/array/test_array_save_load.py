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
import os

import numpy as np
import pytest

from docarray import BaseDoc, DocList, DocVec
from docarray.documents import ImageDoc
from docarray.typing import NdArray, TorchTensor


class MyDoc(BaseDoc):
    embedding: NdArray
    text: str
    image: ImageDoc


@pytest.mark.slow
@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle', 'json-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize('show_progress', [False, True])
@pytest.mark.parametrize('array_cls', [DocList, DocVec])
def test_array_save_load_binary(protocol, compress, tmp_path, show_progress, array_cls):
    tmp_file = os.path.join(tmp_path, 'test')

    da = array_cls[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ]
    )

    da.save_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    da2 = array_cls[MyDoc].load_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    assert len(da2) == 2
    assert len(da) == len(da2)
    for d1, d2 in zip(da, da2):
        assert d1.embedding.tolist() == d2.embedding.tolist()
        assert d1.text == d2.text
        assert d1.image.url == d2.image.url
    assert da[1].image.url is None
    assert da2[1].image.url is None


@pytest.mark.slow
@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle', 'json-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize('show_progress', [False, True])
@pytest.mark.parametrize('to_doc_vec', [True, False])
def test_array_save_load_binary_streaming(
    protocol, compress, tmp_path, show_progress, to_doc_vec
):
    tmp_file = os.path.join(tmp_path, 'test')
    array_cls = DocVec if to_doc_vec else DocList

    da = DocList[MyDoc]()

    def _extend_da(num_docs=100):
        for _ in range(num_docs):
            da.extend(
                [
                    MyDoc(
                        embedding=np.random.rand(3, 2),
                        text='hello',
                        image=ImageDoc(url='aux.png'),
                    ),
                ]
            )

    _extend_da()
    if to_doc_vec:
        da = da.to_doc_vec()

    da.save_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    da_after = array_cls[MyDoc].load_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    for i, doc in enumerate(da_after):
        assert doc.id == da[i].id
        assert doc.text == da[i].text
        assert doc.image.url == da[i].image.url

    assert i == 99


@pytest.mark.parametrize('tensor_type', [NdArray, TorchTensor])
def test_save_load_tensor_type(tensor_type, tmp_path):
    tmp_file = os.path.join(tmp_path, 'test123')

    class MyDoc(BaseDoc):
        embedding: tensor_type
        text: str
        image: ImageDoc

    da = DocVec[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ],
        tensor_type=tensor_type,
    )
    da.save_binary(tmp_file)
    da2 = DocVec[MyDoc].load_binary(tmp_file, tensor_type=tensor_type)
    assert da2.tensor_type == tensor_type
    assert isinstance(da2.embedding, tensor_type)
