from __future__ import annotations

import unittest.mock as mock
from collections.abc import Hashable, Iterable, Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

import nrtk.experimental  # noqa: F401 - enable experimental features
from nrtk.impls.perturb_video import FramewisePerturber
from nrtk.interfaces import PerturbImage, VideoFrame
from tests.fakes import FakeImagePerturber
from tests.impls import INPUT_DRONE_VIDEO_FILE_PATH as INPUT_VIDEO_FILE_PATH
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_video.perturber_tests_mixin import PerturbVideoTestsMixin
from tests.impls.perturb_video.test_perturber_utils import FrameRepeater, perturber_assertions
from tests.utils.video_io import read_video

rng = np.random.default_rng()


def _perturb(
    image: np.ndarray,
    boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
    kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:  # pragma: no cover
    return np.copy(image) + 1, deepcopy(boxes)


m_dummy = mock.Mock(spec=PerturbImage)
m_dummy.side_effect = _perturb


@pytest.mark.core
class TestFramewisePerturber(PerturbVideoTestsMixin):
    impl_class = FramewisePerturber

    @override
    def make_perturber(self) -> FramewisePerturber:
        return FramewisePerturber(FakeImagePerturber())

    @override
    def make_frames(self) -> list[VideoFrame]:
        return [
            VideoFrame(image=np.ones((8, 8, 3), dtype=np.uint8), timestamp=0.0),
            VideoFrame(image=np.ones((8, 8, 3), dtype=np.uint8), timestamp=1.0),
        ]

    @pytest.mark.parametrize(
        ("frames", "expectation"),
        [
            ([], does_not_raise()),
            (
                [
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.uint8), timestamp=0.0),
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.uint8), timestamp=0.0),
                ],
                pytest.raises(ValueError, match=r"Sequential frame timestamps are not increasing"),
            ),
            (
                [
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.uint8), timestamp=1.0),
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.uint8), timestamp=0.0),
                ],
                pytest.raises(ValueError, match=r"Sequential frame timestamps are not increasing"),
            ),
            (
                [
                    VideoFrame(image=np.ones((1, 1, 1, 3), dtype=np.uint8), timestamp=0.0),
                ],
                pytest.raises(ValueError, match=r"Frame image must have 2 or 3 dimensions"),
            ),
            (
                [
                    VideoFrame(image=np.ones((3,), dtype=np.uint8), timestamp=0.0),
                ],
                pytest.raises(ValueError, match=r"Frame image must have 2 or 3 dimensions"),
            ),
            (
                [
                    VideoFrame(image=np.ones((0, 1, 3), dtype=np.uint8), timestamp=0.0),
                ],
                pytest.raises(ValueError, match=r"Empty frame image"),
            ),
            (
                [
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.uint8), timestamp=0.0),
                    VideoFrame(image=np.ones((1, 2, 3), dtype=np.uint8), timestamp=1.0),
                ],
                pytest.raises(ValueError, match=r"Sequential frame images do not have the same shape"),
            ),
            (
                [
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.uint8), timestamp=0.0),
                    VideoFrame(image=np.ones((1, 1, 3), dtype=np.float32), timestamp=1.0),
                ],
                pytest.raises(ValueError, match=r"Sequential frame images do not have the same data type"),
            ),
        ],
    )
    def test_invalid_value(self, frames: Iterator[VideoFrame], expectation: AbstractContextManager) -> None:
        inst = FramewisePerturber(FakeImagePerturber())
        with expectation:
            for _ in perturber_assertions(perturb=inst, frames=frames):
                pass

    @pytest.mark.parametrize(
        ("frames", "expected"),
        [
            (
                [
                    VideoFrame(
                        image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)[:, :, None],
                        timestamp=0.0,
                        boxes=[(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(3, 1)), {"meta": 1})],
                    ),
                ],
                [
                    VideoFrame(
                        image=np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=np.uint8)[:, :, None],
                        timestamp=0.0,
                        boxes=[(AxisAlignedBoundingBox(min_vertex=(1, 0), max_vertex=(3, 1)), {"meta": 1})],
                    ),
                ],
            ),
        ],
    )
    def test_consistency(self, frames: Iterator[VideoFrame], expected: Iterator[VideoFrame]) -> None:
        inst = FramewisePerturber(m_dummy)
        for _ in perturber_assertions(perturb=inst.perturb, frames=frames, expecteds=iter(expected)):
            pass
        for _ in perturber_assertions(perturb=inst, frames=frames, expecteds=iter(expected)):
            pass

    @pytest.mark.parametrize(
        ("frames"),
        [
            FrameRepeater(
                frame=VideoFrame(
                    image=rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8),
                    timestamp=0.0,
                ),
                n=5,
            ),
            FrameRepeater(frame=VideoFrame(image=np.ones((64, 64, 1), dtype=np.float32), timestamp=0.0), n=3),
            FrameRepeater(frame=VideoFrame(image=np.ones((128, 128, 3), dtype=np.float64), timestamp=0.0), n=15),
        ],
    )
    def test_reproducibility(self, frames: Iterator[VideoFrame]) -> None:
        inst = FramewisePerturber(m_dummy)
        perturbed_frames = list(perturber_assertions(perturb=inst, frames=frames))

        inst = FramewisePerturber(m_dummy)
        for _ in perturber_assertions(perturb=inst, frames=frames, expecteds=iter(perturbed_frames)):
            pass

        inst = FramewisePerturber(m_dummy)
        for _ in perturber_assertions(perturb=inst.perturb, frames=frames, expecteds=iter(perturbed_frames)):
            pass

    def test_identity_operation(self) -> None:
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        frames = FrameRepeater(frame=VideoFrame(image=image, timestamp=0.0), n=30)
        inst = FramewisePerturber()
        for _ in perturber_assertions(
            perturb=inst.perturb,
            frames=frames,
            expecteds=frames,
        ):
            pass

    def test_simultaneous_perturb(self) -> None:
        frames1 = FrameRepeater(
            frame=VideoFrame(image=rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), timestamp=0.0),
            n=30,
        )
        frames2 = FrameRepeater(
            frame=VideoFrame(image=rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), timestamp=0.0),
            n=30,
        )

        inst = FramewisePerturber(FakeImagePerturber())

        output1 = perturber_assertions(perturb=inst, frames=frames1, expecteds=frames1)
        next(output1)

        with pytest.raises(RuntimeError, match=r"Cannot call perturb\(\) while previous perturb\(\) is incomplete"):
            next(perturber_assertions(perturb=inst, frames=frames2))

        for _ in zip(range(29), output1, strict=False):
            pass

        with pytest.raises(RuntimeError, match=r"Cannot call perturb\(\) while previous perturb\(\) is incomplete"):
            next(perturber_assertions(perturb=inst, frames=frames2))

        with pytest.raises(StopIteration):
            next(output1)

        for _ in perturber_assertions(perturb=inst, frames=frames2, expecteds=frames2):
            pass

    def test_regression(self, lossless_mp4_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        frames = read_video(INPUT_VIDEO_FILE_PATH)
        inst = FramewisePerturber(m_dummy)
        out_video = perturber_assertions(perturb=inst, frames=frames)
        lossless_mp4_snapshot.assert_match(out_video)

    @pytest.mark.parametrize(
        ("param1", "param2"),
        [
            (0, 0),
            (1, 2),
        ],
    )
    def test_configuration(
        self,
        param1: float,
        param2: float,
    ) -> None:
        inst = FramewisePerturber(FakeImagePerturber(param1=param1, param2=param2))
        for i in configuration_test_helper(inst):
            assert isinstance(i.frame_perturber, FakeImagePerturber)
            assert i.frame_perturber.param1 == param1
            assert i.frame_perturber.param2 == param2
