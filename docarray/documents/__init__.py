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
from docarray.documents.audio import AudioDoc
from docarray.documents.image import ImageDoc
from docarray.documents.mesh import Mesh3D, VerticesAndFaces
from docarray.documents.point_cloud import PointCloud3D, PointsAndColors
from docarray.documents.text import TextDoc
from docarray.documents.video import VideoDoc

__all__ = [
    'TextDoc',
    'ImageDoc',
    'AudioDoc',
    'Mesh3D',
    'VerticesAndFaces',
    'PointCloud3D',
    'PointsAndColors',
    'VideoDoc',
]
