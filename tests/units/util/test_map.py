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
from typing import Generator, Optional

import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.typing import ImageUrl, NdArray
from docarray.utils.map import map_docs, map_docs_batched
from tests.units.typing.test_bytes import IMAGE_PATHS

N_DOCS = 2


def load_from_doc(d: ImageDoc) -> ImageDoc:
    if d.url is not None:
        d.tensor = d.url.load()
    return d


@pytest.fixture()
def da():
    da = DocList[ImageDoc]([ImageDoc(url=IMAGE_PATHS['png']) for _ in range(N_DOCS)])
    return da


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_map(da, backend):
    for tensor in da.tensor:
        assert tensor is None

    docs = list(map_docs(docs=da, func=load_from_doc, backend=backend))

    assert len(docs) == N_DOCS
    for doc in docs:
        assert doc.tensor is not None


def test_map_multiprocessing_lambda_func_raise_exception(da):
    with pytest.raises(ValueError, match='Multiprocessing does not allow'):
        list(map_docs(docs=da, func=lambda x: x, backend='process'))


def test_map_multiprocessing_local_func_raise_exception(da):
    def local_func(x):
        return x

    with pytest.raises(ValueError, match='Multiprocessing does not allow'):
        list(map_docs(docs=da, func=local_func, backend='process'))


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_check_order(backend):
    da = DocList[ImageDoc]([ImageDoc(id=str(i)) for i in range(N_DOCS)])

    docs = list(map_docs(docs=da, func=load_from_doc, backend=backend))

    assert len(docs) == N_DOCS
    for i, doc in enumerate(docs):
        assert doc.id == str(i)


def load_from_da(da: DocList) -> DocList:
    for doc in da:
        doc.tensor = doc.url.load()
    return da


class MyImage(BaseDoc):
    tensor: Optional[NdArray] = None
    url: ImageUrl


@pytest.mark.slow
@pytest.mark.parametrize('n_docs,batch_size', [(10, 5), (10, 8)])
@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_map_docs_batched(n_docs, batch_size, backend):
    da = DocList[MyImage]([MyImage(url=IMAGE_PATHS['png']) for _ in range(n_docs)])
    it = map_docs_batched(
        docs=da, func=load_from_da, batch_size=batch_size, backend=backend
    )
    assert isinstance(it, Generator)

    for batch in it:
        assert isinstance(batch, DocList[MyImage])
