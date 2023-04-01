import torch

from docarray.computation.numpy_backend import NumpyCompBackend
from docarray.computation.torch_backend import TorchCompBackend

np_metrics = NumpyCompBackend.Metrics
torch_metrics = TorchCompBackend.Metrics


def test_cosine_sim_compare():
    a = torch.rand(128)
    b = torch.rand(128)
    torch.testing.assert_close(
        torch_metrics.cosine_sim(a, b),
        torch.from_numpy(np_metrics.cosine_sim(a.numpy(), b.numpy())),
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    torch.testing.assert_close(
        torch_metrics.cosine_sim(a, b),
        torch.from_numpy(np_metrics.cosine_sim(a.numpy(), b.numpy())),
    )


def test_euclidean_dist_compare():
    a = torch.rand(128)
    b = torch.rand(128)
    torch.testing.assert_close(
        torch_metrics.euclidean_dist(a, b),
        torch.from_numpy(np_metrics.euclidean_dist(a.numpy(), b.numpy())).to(
            torch.float32
        ),
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    torch.testing.assert_close(
        torch_metrics.euclidean_dist(a, b),
        torch.from_numpy(np_metrics.euclidean_dist(a.numpy(), b.numpy())),
    )


def test_sqeuclidean_dist_compare():
    a = torch.rand(128)
    b = torch.rand(128)
    torch.testing.assert_close(
        torch_metrics.sqeuclidean_dist(a, b),
        torch.from_numpy(np_metrics.sqeuclidean_dist(a.numpy(), b.numpy())).to(
            torch.float32
        ),
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    torch.testing.assert_close(
        torch_metrics.sqeuclidean_dist(a, b),
        torch.from_numpy(np_metrics.sqeuclidean_dist(a.numpy(), b.numpy())),
    )
