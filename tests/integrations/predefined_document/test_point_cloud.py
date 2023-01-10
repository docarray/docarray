import numpy as np
import pytest

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
