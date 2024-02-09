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
import pytest
from pydantic import parse_obj_as

from docarray.typing import NdArray, TorchTensor

try:
    from docarray.typing import TensorFlowTensor
except (ImportError, TypeError):
    pass


@pytest.mark.tensorflow
def test_coercion_behavior():
    t_np = parse_obj_as(NdArray[128], np.zeros(128))
    t_th = parse_obj_as(TorchTensor[128], np.zeros(128))
    t_tf = parse_obj_as(TensorFlowTensor[128], np.zeros(128))

    assert isinstance(t_np, NdArray[128])
    assert not isinstance(t_np, TensorFlowTensor[128])
    assert not isinstance(t_np, TorchTensor[128])

    assert isinstance(t_th, TorchTensor[128])
    assert not isinstance(t_th, NdArray[128])
    assert not isinstance(t_th, TensorFlowTensor[128])

    assert isinstance(t_tf, TensorFlowTensor[128])
    assert not isinstance(t_tf, TorchTensor[128])
    assert not isinstance(t_tf, NdArray[128])
