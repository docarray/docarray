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
import torch

from docarray.computation.torch_backend import TorchCompBackend

metrics = TorchCompBackend.Metrics


def test_cosine_sim_torch():
    a = torch.rand(128)
    b = torch.rand(128)
    assert metrics.cosine_sim(a, b).shape == (1,)
    assert metrics.cosine_sim(a, b) == metrics.cosine_sim(b, a)
    torch.testing.assert_close(metrics.cosine_sim(a, a), torch.ones(1))

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    assert metrics.cosine_sim(a, b).shape == (10, 5)
    assert metrics.cosine_sim(b, a).shape == (5, 10)
    diag_dists = torch.diagonal(metrics.cosine_sim(b, b))  # self-comparisons
    torch.testing.assert_allclose(diag_dists, torch.ones(5))


def test_euclidean_dist_torch():
    a = torch.rand(128)
    b = torch.rand(128)
    assert metrics.euclidean_dist(a, b).shape == (1,)
    assert metrics.euclidean_dist(a, b) == metrics.euclidean_dist(b, a)
    torch.testing.assert_close(metrics.euclidean_dist(a, a), torch.zeros(1))

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    assert metrics.euclidean_dist(a, b).shape == (10, 5)
    assert metrics.euclidean_dist(b, a).shape == (5, 10)
    diag_dists = torch.diagonal(metrics.euclidean_dist(b, b))  # self-comparisons
    torch.testing.assert_allclose(diag_dists, torch.zeros(5))

    a = torch.tensor([0.0, 2.0, 0.0])
    b = torch.tensor([0.0, 0.0, 2.0])
    desired_output_singleton = torch.sqrt(torch.tensor([2.0**2.0 + 2.0**2.0]))
    torch.testing.assert_close(metrics.euclidean_dist(a, b), desired_output_singleton)

    a = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    desired_output_singleton = torch.tensor(
        [[desired_output_singleton.item(), 0.0], [0.0, desired_output_singleton.item()]]
    )
    torch.testing.assert_close(metrics.euclidean_dist(a, b), desired_output_singleton)


def test_sqeuclidean_dist_torch():
    a = torch.rand(128)
    b = torch.rand(128)
    assert metrics.sqeuclidean_dist(a, b).shape == (1,)
    torch.testing.assert_close(
        metrics.sqeuclidean_dist(a, b),
        metrics.euclidean_dist(a, b) ** 2,
    )

    a = torch.rand(10, 3)
    b = torch.rand(5, 3)
    assert metrics.sqeuclidean_dist(a, b).shape == (10, 5)
    torch.testing.assert_close(
        metrics.sqeuclidean_dist(a, b),
        metrics.euclidean_dist(a, b) ** 2,
    )
