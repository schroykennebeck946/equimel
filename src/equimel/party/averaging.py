from __future__ import annotations

from equimel.stores.notation import StateDict


class DenseWeightAverager:
    def __init__(self, period: int, warmup: int) -> None:
        self.period = max(1, period)
        self.warmup = max(0, warmup)
        self._running: StateDict = {}
        self._count = 0
        self._best = float("inf")
        self._window_open = False

    def observe(self, step: int, state: StateDict, validation_loss: float) -> None:
        if step < self.warmup:
            self._best = min(self._best, validation_loss)
            return
        if not self._window_open and validation_loss <= self._best:
            self._best = validation_loss
        if validation_loss > self._best:
            self._window_open = True
        if not self._window_open or step % self.period != 0:
            return
        self._accumulate(state)

    def _accumulate(self, state: StateDict) -> None:
        if not self._running:
            self._running = {key: value.detach().clone().float() for key, value in state.items()}
            self._count = 1
            return
        self._count += 1
        for key, value in state.items():
            self._running[key] += (value.detach().float() - self._running[key]) / self._count

    def force(self, state: StateDict) -> None:
        self._accumulate(state)

    def has_average(self) -> bool:
        return self._count > 0

    def consolidate(self, reference: StateDict) -> StateDict:
        if not self.has_average():
            return {key: value.detach().clone() for key, value in reference.items()}
        merged: StateDict = {}
        for key, value in reference.items():
            averaged = self._running.get(key)
            merged[key] = averaged.to(value.dtype) if averaged is not None else value.clone()
        return merged
