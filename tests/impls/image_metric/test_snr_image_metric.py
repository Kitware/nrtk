import re
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nrtk.impls.image_metric.snr_image_metric import SNRImageMetric, _signal_to_noise

from .image_metric_utils import image_metric_assertions


class TestSNRImageMetric:
    """This class contains the unit tests for the functionality of the SNRImageMetric impl."""

    @pytest.mark.parametrize(
        ("img_1", "img_2", "additional_params", "expectation"),
        [
            (  # single random input image, None, None
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                None,
                does_not_raise(),
            ),
            (  # single random input image with ndim != 3, None, None
                np.random.randint(0, 255, (256, 256), dtype=np.uint8),
                None,
                None,
                pytest.raises(
                    ValueError,
                    match="Incorrect number of dimensions on input image! Expected ndim == 3.",
                ),
            ),
            (  # single random input image with height <= 0, None, None
                np.random.randint(0, 255, (0, 256, 3), dtype=np.uint8),
                None,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid image height! Image height is <= 0.",
                ),
            ),
            (  # single random input image with width <= 0, None, None
                np.random.randint(0, 255, (256, 0, 3), dtype=np.uint8),
                None,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid image width! Image width is <= 0.",
                ),
            ),
            (  # single random input image with number of channels = 2, None, None
                np.random.randint(0, 255, (256, 256, 2), dtype=np.uint8),
                None,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid number of channels on input image! Expected 1 or 3.",
                ),
            ),
            (  # two random input images, None
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                pytest.raises(
                    ValueError,
                    match="Incorrect number of arguments. Computing SNR can only be done on a single image.",
                ),
            ),
            (  # single random input image, None, a dict containing just "axis" = None
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"axis": None},
                does_not_raise(),
            ),
            (  # single random input image, None, a dict containing just "axis" = 0
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"axis": 0},
                does_not_raise(),
            ),
            (  # single random input image, None, a dict containing just "axis" = 1
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"axis": 1},
                does_not_raise(),
            ),
            (  # single random input image, None, a dict containing just "axis" = (0,1)
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"axis": (0, 1)},
                does_not_raise(),
            ),
            (  # single random input image, None, a dict containing "axis" = 0 and "ddof" = 1
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"axis": 0, "ddof": 1},
                does_not_raise(),
            ),
            (  # single random input image, None, a dict containing just "axis" = 2
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"axis": 2},
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid axis parameter! Valid axis parameters are: None, 0, 1, (0,1)."
                    ),
                ),
            ),
            (  # single random input image, None, a dict containing "ddof" = -1
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"ddof": -1},
                pytest.raises(
                    ValueError,
                    match="Invalid ddof value! ddof must be non-negative.",
                ),
            ),
            (  # single random input image, None, a dict containing "ddof" = 100000000
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                None,
                {"ddof": 100000000},
                pytest.raises(
                    ValueError,
                    match=r"Invalid ddof value! ddof must be at most \d+, depending on the value of axis.",
                ),
            ),
        ],
        ids=[
            "Single random input image",
            "Incorrect ndim",
            "Invalid height",
            "Invalid width",
            "Invalid number of channels",
            "Two random input images",
            "axis param = None",
            "axis param =  0",
            "axis param = 1",
            "axis param = (0,1)",
            "axis param = 0 and ddof param = 1",
            "axis param = 2",
            "ddof param = -1",
            "ddof param = 100000000",
        ],
    )
    def test_compute_snr(
        self,
        img_1: np.ndarray,
        img_2: Optional[np.ndarray],
        additional_params: Optional[Dict[str, Any]],
        expectation: ContextManager,
    ) -> None:
        """Test computeSNR with various random image inputs and parameters."""
        with expectation:
            inst = SNRImageMetric()

            # Test ComputeSNR interface directly
            compute_snr = image_metric_assertions(
                computation=inst.compute,
                img_1=img_1,
                img_2=img_2,
                additional_params=additional_params,
            )

            # Test callable
            callable_snr = image_metric_assertions(
                computation=inst,
                img_1=img_1,
                img_2=img_2,
                additional_params=additional_params,
            )

            assert compute_snr == callable_snr

    def test_signal_to_noise_calculation(self) -> None:
        """Tests that the _signal_to_noise calculation is consistent.

        When run with the following parameters, has expected value of 1.11803399.
            image = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
            axis = None
            ddof = 0
        """
        EXPECTED_SNR = 1.11803399  # noqa: N806

        image = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
        axis = None
        ddof = 0
        calculated_snr = _signal_to_noise(image, axis, ddof)
        assert calculated_snr == pytest.approx(EXPECTED_SNR)

    @patch("nrtk.impls.image_metric.snr_image_metric._signal_to_noise")
    def test_signal_to_noise(self, mock_signal_to_noise: MagicMock) -> None:
        """Tests that the inputs are properly received inside of _signal_to_noise()."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        axis = None
        ddof = 0

        inst = SNRImageMetric()

        inst.compute(image, additional_params={"axis": axis, "ddof": ddof})

        # Assert that _signal_to_noise was called with the correct arguments
        mock_signal_to_noise.assert_called_once_with(img=image, axis=axis, ddof=ddof)
