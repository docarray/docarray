import numpy as np
import pytest

from docarray.math.helper import minmax_normalize


@pytest.mark.parametrize(
    'array,t_range,x_range,result',
    [
        (np.array([0, 1, 2, 3, 4, 5]), (0, 10), None, np.array([0, 2, 4, 6, 8, 10])),
        (np.array([[0, 1], [0, 1]]), (0, 10), None, np.array([[0, 10], [0, 10]])),
        (np.array([0, 1, 2, 3, 4, 5]), (0, 10), (0, 10), np.array([0, 1, 2, 3, 4, 5])),
    ],
)
def test_minmax_normalize(array, t_range, x_range, result):
    output = minmax_normalize(x=array, t_range=t_range, x_range=x_range)
    assert np.allclose(output, result)
