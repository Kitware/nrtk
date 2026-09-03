"""Tests for NumpyRandomPerturbVideo base class."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

import numpy as np
import pytest
from typing_extensions import override

from nrtk.impls.perturb_video._base.numpy_random_perturb_video import NumpyRandomPerturbVideo
from nrtk.interfaces import VideoFrame
from nrtk.interfaces._perturb_video import _perturb_guard
from tests.interfaces.test_random_perturb_video import TestRandomPerturbVideo


class _ConcreteNumpyRandomPerturbVideo(NumpyRandomPerturbVideo):
    """Minimal concrete subclass for testing NumpyRandomPerturbVideo."""

    @override
    @_perturb_guard
    def perturb(
        self,
        *,
        frames: Iterator[VideoFrame],
        **additional_params: Any,
    ) -> Generator[VideoFrame, None, None]:
        yield from frames


@pytest.mark.core
class TestNumpyRandomPerturbVideo(TestRandomPerturbVideo):
    @override
    def make_perturber(self, seed: int | None = None) -> NumpyRandomPerturbVideo:
        return _ConcreteNumpyRandomPerturbVideo(seed=seed)

    def test_set_seed_called(self) -> None:
        inst = self.make_perturber(seed=42)
        assert isinstance(inst._rng, np.random.Generator)

    def test_rng_is_generator(self) -> None:
        inst = self.make_perturber(seed=42)
        assert isinstance(inst._rng, np.random.Generator)

    def test_seeded_deterministic(self) -> None:
        inst1 = self.make_perturber(seed=42)
        inst2 = self.make_perturber(seed=42)
        val1 = inst1._rng.random()
        val2 = inst2._rng.random()
        assert val1 == val2

    def test_unseeded_valid(self) -> None:
        inst = self.make_perturber()
        assert isinstance(inst._rng, np.random.Generator)
        val = inst._rng.random()
        assert 0.0 <= val < 1.0

    def test_set_seed_reinitializes(self) -> None:
        inst = self.make_perturber(seed=42)
        val1 = inst._rng.random()
        inst._set_seed()
        val2 = inst._rng.random()
        assert val1 == val2
