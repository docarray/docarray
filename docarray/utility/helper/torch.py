from typing import Optional, Tuple

import torch


def top_k(
    values: 'torch.tensor',
    k: int,
    descending: bool = False,
    device: Optional[str] = None,
) -> Tuple['torch.tensor', 'torch.tensor']:
    """Retrieves the top k smallest values in `values`,
    and returns them alongside their indices in the input `values`.
    Can also be used to retrieve the top k largest values,
    by setting the `descending` flag.

    :param values: Torch tensor of values to rank.
        Should be of shape (n_queries, n_values_per_query).
        Inputs of shape (n_values_per_query,) will be expanded
        to (1, n_values_per_query).
    :param k: number of values to retrieve
    :param descending: retrieve largest values instead of smallest values
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :return: Tuple containing the retrieved values, and their indices.
        Both ar of shape (n_queries, k)
    """
    if device is not None:
        values = values.to(device)
    if len(values.shape) <= 1:
        values = values.view(1, -1)
    len_values = values.shape[-1] if len(values.shape) > 1 else len(values)
    k = min(k, len_values)
    return torch.topk(input=values, k=k, largest=descending, sorted=True, dim=-1)
