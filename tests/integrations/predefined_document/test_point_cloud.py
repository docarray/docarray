import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import PointCloud3D
from tests import TOYDATA_DIR

LOCAL_OBJ_FILE = str(TOYDATA_DIR / 'tetrahedron.obj')
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_OBJ_FILE, REMOTE_OBJ_FILE])
def test_point_cloud(file_url):
    print(f"file_url = {file_url}")
    point_cloud = PointCloud3D(url=file_url)

    point_cloud.tensor = point_cloud.url.load(samples=100)

    assert isinstance(point_cloud.tensor, np.ndarray)


def test_point_cloud_np():
    image = parse_obj_as(PointCloud3D, np.zeros((10, 10, 3)))
    assert (image.tensor == np.zeros((10, 10, 3))).all()


def test_point_cloud_torch():
    image = parse_obj_as(PointCloud3D, torch.zeros(10, 10, 3))
    assert (image.tensor == torch.zeros(10, 10, 3)).all()


def test_point_cloud_shortcut_doc():
    class MyDoc(BaseDocument):
        image: PointCloud3D
        image2: PointCloud3D
        image3: PointCloud3D

    doc = MyDoc(
        image='http://myurl.ply',
        image2=np.zeros((10, 10, 3)),
        image3=torch.zeros(10, 10, 3),
    )
    assert doc.image.url == 'http://myurl.ply'
    assert (doc.image2.tensor == np.zeros((10, 10, 3))).all()
    assert (doc.image3.tensor == torch.zeros(10, 10, 3)).all()
