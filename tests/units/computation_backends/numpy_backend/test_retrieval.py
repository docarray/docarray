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

from docarray.computation.numpy_backend import NumpyCompBackend


def test_topk_numpy():
    top_k = NumpyCompBackend.Retrieval.top_k

    a = np.array([1, 4, 2, 7, 4, 9, 2])
    vals, indices = top_k(a, 3)
    assert vals.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (vals.squeeze() == np.array([1, 2, 2])).all()
    assert (indices.squeeze() == np.array([0, 2, 6])).all() or (
        indices.squeeze() == np.array([0, 6, 2])
    ).all()

    a = np.array([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]])
    vals, indices = top_k(a, 3)
    assert vals.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert (vals[0] == np.array([1, 2, 2])).all()
    assert (indices[0] == np.array([0, 2, 6])).all() or (
        indices[0] == np.array([0, 6, 2])
    ).all()
    assert (vals[1] == np.array([2, 3, 4])).all()
    assert (indices[1] == np.array([2, 4, 6])).all()
