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

from docarray.base_doc.doc import BaseDoc
from docarray.documents import Mesh3D
from tests import TOYDATA_DIR

LOCAL_OBJ_FILE = str(TOYDATA_DIR / 'tetrahedron.obj')
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'

pytestmark = [pytest.mark.mesh]


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_OBJ_FILE, REMOTE_OBJ_FILE])
def test_mesh(file_url: str):
    mesh = Mesh3D(url=file_url)

    mesh.tensors = mesh.url.load()

    assert isinstance(mesh.tensors.vertices, np.ndarray)
    assert isinstance(mesh.tensors.faces, np.ndarray)


def test_str_init():
    t = parse_obj_as(Mesh3D, 'http://hello.ply')
    assert t.url == 'http://hello.ply'


def test_doc():
    class MyDoc(BaseDoc):
        mesh1: Mesh3D
        mesh2: Mesh3D

    doc = MyDoc(mesh1='http://hello.ply', mesh2=Mesh3D(url='http://hello.ply'))

    assert doc.mesh1.url == 'http://hello.ply'
    assert doc.mesh2.url == 'http://hello.ply'
