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
from typing import TypeVar

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode

T = TypeVar('T', bound='ImageTorchTensor')


@_register_proto(proto_type_name='image_torch_tensor')
class ImageTorchTensor(AbstractImageTensor, TorchTensor, metaclass=metaTorchAndNode):
    """
    Subclass of [`TorchTensor`][docarray.typing.TorchTensor], to represent an image tensor.
    Adds image-specific features to the tensor.
    For instance the ability convert the tensor back to
    [`ImageBytes`][docarray.typing.ImageBytes] which are
    optimized to send over the wire.


    ---

    ```python
    from typing import Optional

    from docarray import BaseDoc
    from docarray.typing import ImageBytes, ImageTorchTensor, ImageUrl


    class MyImageDoc(BaseDoc):
        title: str
        tensor: Optional[ImageTorchTensor] = None
        url: Optional[ImageUrl] = None
        bytes: Optional[ImageBytes] = None


    doc = MyImageDoc(
        title='my_second_image_doc',
        url="https://upload.wikimedia.org/wikipedia/commons/8/80/"
        "Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg",
    )

    doc.tensor = doc.url.load()
    doc.bytes = doc.tensor.to_bytes()
    ```

    ---
    """

    ...
