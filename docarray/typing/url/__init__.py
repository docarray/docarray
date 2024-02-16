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
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.audio_url import AudioUrl
from docarray.typing.url.image_url import ImageUrl
from docarray.typing.url.text_url import TextUrl
from docarray.typing.url.url_3d.mesh_url import Mesh3DUrl
from docarray.typing.url.url_3d.point_cloud_url import PointCloud3DUrl
from docarray.typing.url.video_url import VideoUrl

__all__ = [
    'ImageUrl',
    'AudioUrl',
    'AnyUrl',
    'TextUrl',
    'Mesh3DUrl',
    'PointCloud3DUrl',
    'VideoUrl',
]
