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
from typing import Any  # noqa: F401

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor

tensorflow_base = type(TensorFlowTensor)  # type: Any
embedding_base = type(EmbeddingMixin)  # type: Any


class metaTensorFlowAndEmbedding(tensorflow_base, embedding_base):
    pass


@_register_proto(proto_type_name='tensorflow_embedding')
class TensorFlowEmbedding(
    TensorFlowTensor, EmbeddingMixin, metaclass=metaTensorFlowAndEmbedding
):
    alternative_type = TensorFlowTensor
