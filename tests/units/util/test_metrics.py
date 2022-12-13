import numpy as np
import torch

import docarray.utility.math.metrics.numpy as np_distances
import docarray.utility.math.metrics.torch as torch_distances


def test_cosine_sim_torch():
    a = torch.rand(128)
    b = torch.rand(128)
    assert torch_distances.cosine_sim(a, b).shape == (1,)
    assert torch_distances.cosine_sim(a, b) == torch_distances.cosine_sim(b, a)
    torch.testing.assert_close(torch_distances.cosine_sim(a, a), torch.ones(1))

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    assert torch_distances.cosine_sim(a, b).shape == (10, 5)
    assert torch_distances.cosine_sim(b, a).shape == (5, 10)
    diag_dists = torch.diagonal(torch_distances.cosine_sim(b, b))  # self-comparisons
    torch.testing.assert_allclose(diag_dists, torch.ones(5))


def test_cosine_sim_np():
    a = np.random.rand(128)
    b = np.random.rand(128)
    assert np_distances.cosine_sim(a, b).shape == (1,)
    assert np_distances.cosine_sim(a, b) == np_distances.cosine_sim(b, a)
    np.testing.assert_array_almost_equal(np_distances.cosine_sim(a, a), np.ones((1,)))

    a = np.random.rand(10, 3)
    b = np.random.rand(5, 3)
    assert np_distances.cosine_sim(a, b).shape == (10, 5)
    assert np_distances.cosine_sim(b, a).shape == (5, 10)
    diag_dists = np.diagonal(np_distances.cosine_sim(b, b))  # self-comparisons
    np.testing.assert_array_almost_equal(diag_dists, np.ones((5,)))


def test_cosine_sim_compare():
    a = torch.rand(128)
    b = torch.rand(128)
    torch.testing.assert_close(
        torch_distances.cosine_sim(a, b),
        torch.from_numpy(np_distances.cosine_sim(a.numpy(), b.numpy())),
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    torch.testing.assert_close(
        torch_distances.cosine_sim(a, b),
        torch.from_numpy(np_distances.cosine_sim(a.numpy(), b.numpy())),
    )


def test_euclidean_dist_torch():
    a = torch.rand(128)
    b = torch.rand(128)
    assert torch_distances.euclidean_dist(a, b).shape == (1,)
    assert torch_distances.euclidean_dist(a, b) == torch_distances.euclidean_dist(b, a)
    torch.testing.assert_close(torch_distances.euclidean_dist(a, a), torch.zeros(1))

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    assert torch_distances.euclidean_dist(a, b).shape == (10, 5)
    assert torch_distances.euclidean_dist(b, a).shape == (5, 10)
    diag_dists = torch.diagonal(
        torch_distances.euclidean_dist(b, b)
    )  # self-comparisons
    torch.testing.assert_allclose(diag_dists, torch.zeros(5))

    a = torch.tensor([0.0, 2.0, 0.0])
    b = torch.tensor([0.0, 0.0, 2.0])
    desired_output_singleton = torch.sqrt(torch.tensor([2.0**2.0 + 2.0**2.0]))
    torch.testing.assert_close(
        torch_distances.euclidean_dist(a, b), desired_output_singleton
    )

    a = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    desired_output_singleton = torch.tensor(
        [[desired_output_singleton.item(), 0.0], [0.0, desired_output_singleton.item()]]
    )
    torch.testing.assert_close(
        torch_distances.euclidean_dist(a, b), desired_output_singleton
    )


def test_euclidean_dist_np():
    a = np.random.rand(128)
    b = np.random.rand(128)
    assert np_distances.euclidean_dist(a, b).shape == (1,)
    assert np_distances.euclidean_dist(a, b) == np_distances.euclidean_dist(b, a)
    np.testing.assert_array_almost_equal(
        np_distances.euclidean_dist(a, a), np.zeros((1,))
    )

    a = np.random.rand(10, 3)
    b = np.random.rand(5, 3)
    assert np_distances.euclidean_dist(a, b).shape == (10, 5)
    assert np_distances.euclidean_dist(b, a).shape == (5, 10)
    diag_dists = np.diagonal(np_distances.euclidean_dist(b, b))  # self-comparisons
    np.testing.assert_array_almost_equal(diag_dists, np.zeros((5,)))

    a = np.array([0.0, 2.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])
    desired_output_singleton = np.sqrt(np.array([2.0**2.0 + 2.0**2.0]))
    np.testing.assert_array_almost_equal(
        np_distances.euclidean_dist(a, b), desired_output_singleton
    )

    a = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    b = np.array([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    desired_output_singleton = np.array(
        [[desired_output_singleton.item(), 0.0], [0.0, desired_output_singleton.item()]]
    )
    np.testing.assert_array_almost_equal(
        np_distances.euclidean_dist(a, b), desired_output_singleton
    )


def test_euclidean_dist_compare():
    a = torch.rand(128)
    b = torch.rand(128)
    torch.testing.assert_close(
        torch_distances.euclidean_dist(a, b),
        torch.from_numpy(np_distances.euclidean_dist(a.numpy(), b.numpy())).to(
            torch.float32
        ),
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    torch.testing.assert_close(
        torch_distances.euclidean_dist(a, b),
        torch.from_numpy(np_distances.euclidean_dist(a.numpy(), b.numpy())),
    )


def test_sqeuclidean_dist_torch():
    a = torch.rand(128)
    b = torch.rand(128)
    assert torch_distances.sqeuclidean_dist(a, b).shape == (1,)
    torch.testing.assert_close(
        torch_distances.sqeuclidean_dist(a, b),
        torch_distances.euclidean_dist(a, b) ** 2,
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    assert torch_distances.sqeuclidean_dist(a, b).shape == (10, 5)
    torch.testing.assert_close(
        torch_distances.sqeuclidean_dist(a, b),
        torch_distances.euclidean_dist(a, b) ** 2,
    )


def test_sqeuclidea_dist_np():
    a = np.random.rand(128)
    b = np.random.rand(128)
    assert np_distances.sqeuclidean_dist(a, b).shape == (1,)
    np.testing.assert_array_almost_equal(
        np_distances.sqeuclidean_dist(a, b), np_distances.euclidean_dist(a, b) ** 2
    )

    a = np.random.rand(10, 3)
    b = np.random.rand(5, 3)
    assert np_distances.sqeuclidean_dist(a, b).shape == (10, 5)
    np.testing.assert_array_almost_equal(
        np_distances.sqeuclidean_dist(a, b), np_distances.euclidean_dist(a, b) ** 2
    )
