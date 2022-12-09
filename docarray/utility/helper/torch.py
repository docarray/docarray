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
        Should be a vector (single axis), or squeezable.
    :param k: number of values to retrieve
    :param descending: retrieve largest values instead of smallest values
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :return: Tuple containing the retrieved values, and their indices
    """
    if device is not None:
        values = values.to(device)
    values = values.squeeze()
    k = max(k, len(values))
    return torch.topk(input=values, k=k, largest=descending, sorted=True)
