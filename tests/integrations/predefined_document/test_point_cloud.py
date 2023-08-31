import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import PointCloud3D
from docarray.utils._internal.misc import is_tf_available
from docarray.utils._internal.pydantic import is_pydantic_v2
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

LOCAL_OBJ_FILE = str(TOYDATA_DIR / 'tetrahedron.obj')
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_OBJ_FILE, REMOTE_OBJ_FILE])
def test_point_cloud(file_url):
    print(f"file_url = {file_url}")
    point_cloud = PointCloud3D(url=file_url)

    point_cloud.tensors = point_cloud.url.load(samples=100)

    assert isinstance(point_cloud.tensors.points, np.ndarray)


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_point_cloud_np():
    pc = parse_obj_as(PointCloud3D, np.zeros((10, 3)))
    assert (pc.tensors.points == np.zeros((10, 3))).all()


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_point_cloud_torch():
    pc = parse_obj_as(PointCloud3D, torch.zeros(10, 3))
    assert (pc.tensors.points == torch.zeros(10, 3)).all()


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
@pytest.mark.tensorflow
def test_point_cloud_tensorflow():
    pc = parse_obj_as(PointCloud3D, tf.zeros((10, 3)))
    assert tnp.allclose(pc.tensors.points.tensor, tf.zeros((10, 3)))


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_point_cloud_shortcut_doc():
    class MyDoc(BaseDoc):
        pc: PointCloud3D
        pc2: PointCloud3D
        pc3: PointCloud3D

    doc = MyDoc(
        pc='http://myurl.ply',
        pc2=np.zeros((10, 3)),
        pc3=torch.zeros(10, 3),
    )
    assert doc.pc.url == 'http://myurl.ply'
    assert (doc.pc2.tensors.points == np.zeros((10, 3))).all()
    assert (doc.pc3.tensors.points == torch.zeros(10, 3)).all()


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
@pytest.mark.tensorflow
def test_point_cloud_shortcut_doc_tf():
    class MyDoc(BaseDoc):
        pc: PointCloud3D
        pc2: PointCloud3D

    doc = MyDoc(
        pc='http://myurl.ply',
        pc2=tf.zeros((10, 3)),
    )
    assert doc.pc.url == 'http://myurl.ply'
    assert tnp.allclose(doc.pc2.tensors.points.tensor, tf.zeros((10, 3)))
