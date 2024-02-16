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
import os

from pydantic import parse_obj_as

from docarray.typing import ImageBytes, ImageTensor, ImageUrl

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_DATA = os.path.join(CUR_DIR, '..', '..', 'toydata', 'image-data')
IMAGE_PATHS = {
    'png': os.path.join(PATH_TO_IMAGE_DATA, 'so_good.png'),
    'jpg': os.path.join(PATH_TO_IMAGE_DATA, '05984.jpg'),
    'jpeg': os.path.join(PATH_TO_IMAGE_DATA, '05984-2.jpeg'),
}


def test_bytes_load():
    url = parse_obj_as(ImageUrl, IMAGE_PATHS['png'])

    tensor = parse_obj_as(ImageTensor, url.load())

    bytes_ = parse_obj_as(ImageBytes, tensor.to_bytes())

    tensor_new = parse_obj_as(ImageTensor, bytes_.load())

    assert (tensor_new == tensor).all()
