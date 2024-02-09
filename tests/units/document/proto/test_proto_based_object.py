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

from docarray.proto import DocProto, NodeProto
from docarray.typing import NdArray


@pytest.mark.proto
def test_ndarray():
    original_ndarray = np.zeros((3, 224, 224))

    custom_ndarray = NdArray._docarray_from_native(original_ndarray)

    tensor = NdArray.from_protobuf(custom_ndarray.to_protobuf())

    assert (tensor == original_ndarray).all()


@pytest.mark.proto
def test_document_proto_set():
    data = {}

    nested_item1 = NodeProto(text='hello')

    ndarray = NdArray._docarray_from_native(np.zeros((3, 224, 224)))
    nd_proto = ndarray.to_protobuf()

    nested_item2 = NodeProto(ndarray=nd_proto)

    data['a'] = nested_item1
    data['b'] = nested_item2

    DocProto(data=data)
