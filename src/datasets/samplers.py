from typing import Sequence, Iterator

import torch
from torch.utils.data import WeightedRandomSampler


class LimitedStepsWeightedRandomSampler(WeightedRandomSampler):

    def __init__(self, weights: Sequence[float], num_samples: int, n_step: int, replacement: bool = True,
                 generator=None) -> None:
        super().__init__(weights, num_samples, replacement, generator)
        self.step = n_step
        self.idx = 0

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)


class LoaderWrapper:
    def __init__(self, dataloader, n_step):
        self.step = n_step
        self.idx = 0
        self.iter_loader = iter(dataloader)
        self.dataloader = dataloader

    def __iter__(self):
        return self

    def __len__(self):
        return self.step

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx = 0
            self.iter_loader = iter(self.dataloader)
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)
