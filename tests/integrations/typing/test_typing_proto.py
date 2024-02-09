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
import torch

from docarray import BaseDoc
from docarray.base_doc import AnyDoc
from docarray.typing import (
    AnyEmbedding,
    AnyUrl,
    ImageUrl,
    Mesh3DUrl,
    NdArray,
    PointCloud3DUrl,
    TextUrl,
    TorchTensor,
)


@pytest.mark.proto
def test_proto_all_types():
    class Mymmdoc(BaseDoc):
        tensor: NdArray
        torch_tensor: TorchTensor
        embedding: AnyEmbedding
        any_url: AnyUrl
        image_url: ImageUrl
        text_url: TextUrl
        mesh_url: Mesh3DUrl
        point_cloud_url: PointCloud3DUrl

    doc = Mymmdoc(
        tensor=np.zeros((3, 224, 224)),
        torch_tensor=torch.zeros((3, 224, 224)),
        embedding=np.zeros((100, 1)),
        any_url='http://jina.ai',
        image_url='http://jina.ai/bla.jpg',
        text_url='http://jina.ai',
        mesh_url='http://jina.ai/mesh.obj',
        point_cloud_url='http://jina.ai/mesh.obj',
    )

    new_doc = AnyDoc.from_protobuf(doc.to_protobuf())

    for field, value in new_doc:
        if field == 'embedding':
            # embedding is a Union type, not supported by isinstance
            assert isinstance(value, np.ndarray) or isinstance(value, torch.Tensor)
        else:
            assert isinstance(value, doc._get_field_annotation(field))


@pytest.mark.tensorflow
def test_proto_all_types_proto3():
    import tensorflow as tf

    from docarray.typing import TensorFlowTensor

    class Mymmdoc(BaseDoc):
        tensor: NdArray
        torch_tensor: TorchTensor
        tf_tensor: TensorFlowTensor
        embedding: AnyEmbedding
        any_url: AnyUrl
        image_url: ImageUrl
        text_url: TextUrl
        mesh_url: Mesh3DUrl
        point_cloud_url: PointCloud3DUrl

    doc = Mymmdoc(
        tensor=np.zeros((3, 224, 224)),
        torch_tensor=torch.zeros((3, 224, 224)),
        tf_tensor=tf.zeros((3, 224, 224)),
        embedding=np.zeros((100, 1)),
        any_url='http://jina.ai',
        image_url='http://jina.ai/bla.jpg',
        text_url='http://jina.ai/file.txt',
        mesh_url='http://jina.ai/mesh.obj',
        point_cloud_url='http://jina.ai/mesh.obj',
    )

    new_doc = AnyDoc.from_protobuf(doc.to_protobuf())

    for field, value in new_doc:
        if field == 'embedding':
            # embedding is a Union type, not supported by isinstance
            assert isinstance(value, np.ndarray) or isinstance(value, torch.Tensor)
        else:
            assert isinstance(value, doc._get_field_annotation(field))
