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
import pytest

from docarray import BaseDoc, DocList
from docarray.typing import NdArray


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('stack', [False, True])
@pytest.mark.parametrize('batch_size,n_batches', [(16, 7), (10, 10)])
def test_batch(shuffle, stack, batch_size, n_batches):
    class MyDoc(BaseDoc):
        id: int
        tensor: NdArray

    t_shape = (32, 32)
    da = DocList[MyDoc](
        [
            MyDoc(
                id=str(i),
                tensor=np.zeros(t_shape),
            )
            for i in range(100)
        ]
    )
    if stack:
        da = da.to_doc_vec()

    batches = list(da._batch(batch_size=batch_size, shuffle=shuffle))
    assert len(batches) == n_batches

    for i, batch in enumerate(batches):
        if i < n_batches - 1:
            assert len(batch) == batch_size
            if stack:
                assert batch.tensor.shape == (batch_size, *t_shape)
        else:
            assert len(batch) <= batch_size

        non_shuffled_ids = [
            i for i in range(i * batch_size, min((i + 1) * batch_size, len(da)))
        ]
        if not shuffle:
            assert batch.id == non_shuffled_ids
        else:
            assert not (batch.id == non_shuffled_ids)
