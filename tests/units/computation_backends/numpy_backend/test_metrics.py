import numpy as np

from docarray.computation.numpy_backend import NumpyCompBackend

metrics = NumpyCompBackend.Metrics


def test_cosine_sim_np():
    a = np.random.rand(128)
    b = np.random.rand(128)
    assert metrics.cosine_sim(a, b).shape == (1,)
    assert metrics.cosine_sim(a, b) == metrics.cosine_sim(b, a)
    np.testing.assert_array_almost_equal(metrics.cosine_sim(a, a), np.ones((1,)))

    a = np.random.rand(10, 3)
    b = np.random.rand(5, 3)
    assert metrics.cosine_sim(a, b).shape == (10, 5)
    assert metrics.cosine_sim(b, a).shape == (5, 10)
    diag_dists = np.diagonal(metrics.cosine_sim(b, b))  # self-comparisons
    np.testing.assert_array_almost_equal(diag_dists, np.ones((5,)))


def test_euclidean_dist_np():
    a = np.random.rand(128)
    b = np.random.rand(128)
    assert metrics.euclidean_dist(a, b).shape == (1,)
    assert metrics.euclidean_dist(a, b) == metrics.euclidean_dist(b, a)
    np.testing.assert_array_almost_equal(metrics.euclidean_dist(a, a), np.zeros((1,)))

    a = np.random.rand(10, 3)
    b = np.random.rand(5, 3)
    assert metrics.euclidean_dist(a, b).shape == (10, 5)
    assert metrics.euclidean_dist(b, a).shape == (5, 10)
    diag_dists = np.diagonal(metrics.euclidean_dist(b, b))  # self-comparisons
    np.testing.assert_array_almost_equal(diag_dists, np.zeros((5,)))

    a = np.array([0.0, 2.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])
    desired_output_singleton = np.sqrt(np.array([2.0**2.0 + 2.0**2.0]))
    np.testing.assert_array_almost_equal(
        metrics.euclidean_dist(a, b), desired_output_singleton
    )

    a = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    b = np.array([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    desired_output_singleton = np.array(
        [[desired_output_singleton.item(), 0.0], [0.0, desired_output_singleton.item()]]
    )
    np.testing.assert_array_almost_equal(
        metrics.euclidean_dist(a, b), desired_output_singleton
    )


def test_sqeuclidea_dist_np():
    a = np.random.rand(128)
    b = np.random.rand(128)
    assert metrics.sqeuclidean_dist(a, b).shape == (1,)
    np.testing.assert_array_almost_equal(
        metrics.sqeuclidean_dist(a, b), metrics.euclidean_dist(a, b) ** 2
    )

    a = np.random.rand(10, 3)
    b = np.random.rand(5, 3)
    assert metrics.sqeuclidean_dist(a, b).shape == (10, 5)
    np.testing.assert_array_almost_equal(
        metrics.sqeuclidean_dist(a, b), metrics.euclidean_dist(a, b) ** 2
    )
