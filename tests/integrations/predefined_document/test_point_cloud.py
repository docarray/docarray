import numpy as np
import pytest

from docarray import PointCloud3D

REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
def test_point_cloud():

    point_cloud = PointCloud3D(url=REMOTE_OBJ_FILE)

    point_cloud.tensor = point_cloud.url.load(samples=100)

    assert isinstance(point_cloud.tensor, np.ndarray)
