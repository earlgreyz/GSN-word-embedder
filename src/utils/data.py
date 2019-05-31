from typing import Tuple, List

import torch


def split_size(percentage: float, size: int):
    assert percentage >= 0 and percentage <= 1
    N = int(percentage * size)
    M = size - N
    return N, M


def pad_tensor(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    pad_size = list(x.shape)
    pad_size[dim] = pad - x.size(dim)
    return torch.cat([x, torch.zeros(*pad_size)], dim=dim)


Batch = Tuple[torch.Tensor, torch.Tensor]


class PadCollate:
    def __init__(self, dim: int):
        self.dim = dim

    def pad_collate(self, batch: List[Batch]) -> Batch:
        N = max(map(lambda x: x[0].shape[self.dim], batch))
        xs = torch.stack(list(map(lambda x: pad_tensor(x[0], pad=N, dim=self.dim), batch)))
        ys = torch.stack(list(map(lambda x: x[1], batch)))
        return xs, ys

    def __call__(self, batch: List[Batch]) -> Batch:
        return self.pad_collate(batch)
