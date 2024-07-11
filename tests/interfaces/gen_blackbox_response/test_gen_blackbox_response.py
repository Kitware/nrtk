import itertools
from contextlib import nullcontext as does_not_raise
from typing import ContextManager, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

from nrtk.interfaces.gen_blackbox_response import gen_perturber_combinations
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


@pytest.mark.parametrize(
    ("factory_lengths", "expectation"),
    [
        (
            [],
            pytest.raises(
                ValueError,
                match=r"No factories passed in, cannot generate perturbations",
            ),
        ),
        (
            [0],
            pytest.raises(
                ValueError, match=r"Factory passed in with invald length of 0"
            ),
        ),
        ([1], does_not_raise()),
        (
            [1, 0],
            pytest.raises(
                ValueError, match=r"Factory passed in with invald length of 0"
            ),
        ),
        ([1, 1], does_not_raise()),
        ([1, 2], does_not_raise()),
        ([2, 3, 4], does_not_raise()),
        ([10, 15, 2], does_not_raise()),
    ],
)
def test_gen_perturber_combinations(
    factory_lengths: Sequence[int], expectation: ContextManager
) -> None:
    with expectation:
        # Create mock factories
        factories = []
        for factory_length in factory_lengths:
            mock_factory = MagicMock(spec=PerturbImageFactory)  # generate mock factory

            # set mock factory len() value, this is what is used by gen_perturber_combinations()
            # the mock object will have a len of 0 if we do not do this
            mock_factory.__len__.return_value = factory_length
            factories.append(mock_factory)

        result = gen_perturber_combinations(factories)

        # Generate expected output based on the input factory lengths
        expected_output = []

        # create a list of range objects for later multiplication
        # output will look like [range(0,2), range(0,5), etc.]
        factory_length_ranges = [
            range(factory_length) for factory_length in factory_lengths
        ]

        # calculate cartesian product for all of the range objects
        # the * breaks the factory_length_ranges argument into multiple arguments, instead of passing the list as one
        cartesian_product = itertools.product(*factory_length_ranges)

        for combination in cartesian_product:
            expected_output.append(
                list(combination)
            )  # must convert to list because itertools.product returns tuple

        np.testing.assert_equal(result, expected_output)
