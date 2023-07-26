import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise
from smqtk_core.configuration import configuration_test_helper
from typing import Any, Callable, ContextManager, Dict, Optional, Type, Union

from nrtk.impls.perturb_image.skimage.random_noise import (
    GaussianPerturber,
    PepperPerturber,
    SaltAndPepperPerturber,
    SaltPerturber,
    SpecklePerturber,
    _SKImagePerturber
)


def blanket_assertions(
    perturb: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    expected: Optional[np.ndarray] = None
) -> None:
    """
    Test the blanket assertions for perturbers that
    1) Input should remain unchanged
    2) Output should not share memory with input (e.g no clones, etc)
    3) Output should have the same shape as input
    4) Output should have the same dtype as input
    Additionally, if ``expected`` is provided
    5) Output should match expected

    :param perturb: Interface with which to generate the perturbation.
    :param image: Input image as numpy array.
    :param expected: (Optional) Expected return value of the perturbation.
    """
    shape = image.shape
    dtype = image.dtype
    copy = np.copy(image)

    out_image = perturb(image)
    assert np.array_equal(image, copy)
    assert not np.shares_memory(image, out_image)
    assert out_image.shape == shape
    assert out_image.dtype == dtype
    if expected is not None:
        assert np.array_equal(out_image, expected)


def rng_assertions(perturber: Type[_SKImagePerturber], rng: int) -> None:
    """
    Test that output is reproducible if a rng or seed is provided.

    :param perturber: SKImage random_noise perturber class of interest.
    :param rng: Seed value.
    """
    dummy_image_A = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_image_B = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test as seed value
    inst1 = perturber(rng=rng)
    out1A = inst1(dummy_image_A)
    out1B = inst1(dummy_image_B)
    inst2 = perturber(rng=rng)
    out2A = inst2(dummy_image_A)
    out2B = inst2(dummy_image_B)
    assert np.array_equal(out1A, out2A)
    assert np.array_equal(out1B, out2B)

    # Test generator
    inst3 = perturber(rng=np.random.default_rng(rng))
    out3A = inst3(dummy_image_A)
    out3B = inst3(dummy_image_B)
    inst4 = perturber(rng=np.random.default_rng(rng))
    out4A = inst4(dummy_image_A)
    out4B = inst4(dummy_image_B)
    assert np.array_equal(out3A, out4A)
    assert np.array_equal(out3B, out4B)


class TestSaltPerturber:
    @pytest.mark.parametrize("image, rng, amount, expected", [
        (np.zeros((3, 3), dtype=np.uint8), 42, 0.5, np.array([[0, 255, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8))
    ])
    def test_consistency(
        self,
        image: np.ndarray,
        rng: Union[np.random.Generator, int],
        amount: float,
        expected: np.ndarray
    ) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        # Test perturb interface directly
        inst = SaltPerturber(amount=amount, rng=rng)
        blanket_assertions(perturb=inst.perturb, image=image, expected=expected)

        # Test callable
        blanket_assertions(
            perturb=SaltPerturber(amount=amount, rng=rng),
            image=image,
            expected=expected
        )

    @pytest.mark.parametrize("image, expectation", [
        (np.zeros((256, 256, 3), dtype=np.uint8), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.uint8) * 255, does_not_raise()),
        (np.zeros((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.half), does_not_raise()),
        (np.ones((256, 356, 3), dtype=np.csingle),
            pytest.raises(NotImplementedError, match=r"Perturb not implemented for"))
    ])
    def test_no_perturbation(self, image: np.ndarray, expectation: ContextManager) -> None:
        """
        Run a dummy image through the perturber with settings for no
        perturbations, expect to get same image back (quantization errors
        possible). This attempts to isolate perturber implementation code
        from external calls to the extent that is possible.
        """
        with expectation:
            blanket_assertions(perturb=SaltPerturber(amount=0), image=image, expected=image)

    @pytest.mark.parametrize("rng", [42, 12345])
    def test_rng(self, rng: int) -> None:
        """
        Ensure results are reproducible.
        """
        rng_assertions(perturber=SaltPerturber, rng=rng)

    @pytest.mark.parametrize("rng, amount", [
        (42, 0.8),
        (np.random.default_rng(12345), 0.3)
    ])
    def test_configuration(self, rng: Union[np.random.Generator, int], amount: float) -> None:
        """
        Test configuration stability
        """
        inst = SaltPerturber(rng=rng, amount=amount)
        for i in configuration_test_helper(inst):
            assert i.rng == rng
            assert i.amount == amount

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"amount": 0.5}, does_not_raise()),
        ({"amount": 0}, does_not_raise()),
        ({"amount": 1}, does_not_raise()),
        ({"amount": 2.}, pytest.raises(ValueError, match=r"SaltPerturber invalid amount")),
        ({"amount": -3.}, pytest.raises(ValueError, match=r"SaltPerturber invalid amount"))
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            SaltPerturber(**kwargs)


class TestPepperPerturber:
    @pytest.mark.parametrize("image, rng, amount, expected", [
        (np.ones((3, 3), dtype=np.uint8) * 255, 42, 0.5,
            np.array([[255, 0, 255], [255, 0, 255], [255, 255, 0]], dtype=np.uint8))
    ])
    def test_consistency(
        self,
        image: np.ndarray,
        rng: Union[np.random.Generator, int],
        amount: float,
        expected: np.ndarray
    ) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        # Test perturb interface directly
        inst = PepperPerturber(amount=amount, rng=rng)
        blanket_assertions(perturb=inst.perturb, image=image, expected=expected)

        # Test callable
        blanket_assertions(
            perturb=PepperPerturber(amount=amount, rng=rng),
            image=image,
            expected=expected
        )

    @pytest.mark.parametrize("image, expectation", [
        (np.zeros((256, 256, 3), dtype=np.uint8), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.uint8) * 255, does_not_raise()),
        (np.zeros((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.half), does_not_raise()),
        (np.ones((256, 356, 3), dtype=np.csingle),
            pytest.raises(NotImplementedError, match=r"Perturb not implemented for"))
    ])
    def test_no_perturbation(self, image: np.ndarray, expectation: ContextManager) -> None:
        """
        Run a dummy image through the perturber with settings for no
        perturbations, expect to get same image back (quantization errors
        possible). This attempts to isolate perturber implementation code
        from external calls to the extent that is possible.
        """
        with expectation:
            blanket_assertions(perturb=PepperPerturber(amount=0), image=image, expected=image)

    @pytest.mark.parametrize("rng", [42, 12345])
    def test_rng(self, rng: int) -> None:
        """
        Ensure results are reproducible.
        """
        rng_assertions(perturber=PepperPerturber, rng=rng)

    @pytest.mark.parametrize("rng, amount", [
        (42, 0.8),
        (np.random.default_rng(12345), 0.3)
    ])
    def test_configuration(self, rng: Union[np.random.Generator, int], amount: float) -> None:
        """
        Test configuration stability
        """
        inst = PepperPerturber(rng=rng, amount=amount)
        for i in configuration_test_helper(inst):
            assert i.rng == rng
            assert i.amount == amount

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"amount": 0.25}, does_not_raise()),
        ({"amount": 0}, does_not_raise()),
        ({"amount": 1}, does_not_raise()),
        ({"amount": 2.5}, pytest.raises(ValueError, match=r"PepperPerturber invalid amount")),
        ({"amount": -4.2}, pytest.raises(ValueError, match=r"PepperPerturber invalid amount"))
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            PepperPerturber(**kwargs)


class TestSaltAndPepperPerturber:
    @pytest.mark.parametrize("image, rng, amount, salt_vs_pepper, expected", [
        (np.zeros((3, 3), dtype=np.uint8), 42, 0.5, 0.5,
            np.array([[0, 255, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8))
    ])
    def test_consistency(
        self,
        image: np.ndarray,
        rng: Union[np.random.Generator, int],
        amount: float,
        salt_vs_pepper: float,
        expected: np.ndarray
    ) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        # Test perturb interface directly
        inst = SaltAndPepperPerturber(amount=amount, salt_vs_pepper=salt_vs_pepper, rng=rng)
        blanket_assertions(perturb=inst.perturb, image=image, expected=expected)

        # Test callable
        blanket_assertions(
            perturb=SaltAndPepperPerturber(amount=amount, salt_vs_pepper=salt_vs_pepper, rng=rng),
            image=image,
            expected=expected
        )

    @pytest.mark.parametrize("image, expectation", [
        (np.zeros((256, 256, 3), dtype=np.uint8), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.uint8) * 255, does_not_raise()),
        (np.zeros((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.half), does_not_raise()),
        (np.ones((256, 356, 3), dtype=np.csingle),
            pytest.raises(NotImplementedError, match=r"Perturb not implemented for"))
    ])
    def test_no_perturbation(self, image: np.ndarray, expectation: ContextManager) -> None:
        """
        Run a dummy image through the perturber with settings for no
        perturbations, expect to get same image back (quantization errors
        possible). This attempts to isolate perturber implementation code
        from external calls to the extent that is possible.
        """
        with expectation:
            blanket_assertions(perturb=SaltAndPepperPerturber(amount=0), image=image, expected=image)

    @pytest.mark.parametrize("rng", [42, 12345])
    def test_rng(self, rng: int) -> None:
        """
        Ensure results are reproducible.
        """
        rng_assertions(perturber=SaltAndPepperPerturber, rng=rng)

    @pytest.mark.parametrize("rng, amount, salt_vs_pepper", [
        (42, 0.8, 0.25),
        (np.random.default_rng(12345), 0.3, 0.2)
    ])
    def test_configuration(self, rng: Union[np.random.Generator, int], amount: float, salt_vs_pepper: float) -> None:
        """
        Test configuration stability
        """
        inst = SaltAndPepperPerturber(rng=rng, amount=amount, salt_vs_pepper=salt_vs_pepper)
        for i in configuration_test_helper(inst):
            assert i.rng == rng
            assert i.amount == amount
            assert i.salt_vs_pepper == salt_vs_pepper

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"amount": 0.45}, does_not_raise()),
        ({"amount": 0}, does_not_raise()),
        ({"amount": 1}, does_not_raise()),
        ({"amount": 1.2}, pytest.raises(ValueError, match=r"SaltAndPepperPerturber invalid amount")),
        ({"amount": -0.2}, pytest.raises(ValueError, match=r"SaltAndPepperPerturber invalid amount")),
        ({"salt_vs_pepper": 0.2}, does_not_raise()),
        ({"salt_vs_pepper": 0}, does_not_raise()),
        ({"salt_vs_pepper": 1}, does_not_raise()),
        ({"salt_vs_pepper": 5}, pytest.raises(ValueError, match=r"SaltAndPepperPerturber invalid salt_vs_pepper")),
        ({"salt_vs_pepper": -3}, pytest.raises(ValueError, match=r"SaltAndPepperPerturber invalid salt_vs_pepper"))
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            SaltAndPepperPerturber(**kwargs)


class TestGaussianPerturber:
    @pytest.mark.parametrize("image, rng, mean, var, expected", [
        (np.zeros((3, 3), dtype=np.uint8), 42, 0, 0.05, np.array([[17, 0, 43], [54, 0, 0], [7, 0, 0]], dtype=np.uint8))
    ])
    def test_consistency(
        self,
        image: np.ndarray,
        rng: Union[np.random.Generator, int],
        mean: float,
        var: float,
        expected: np.ndarray
    ) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        # Test perturb interface directly
        inst = GaussianPerturber(mean=mean, var=var, rng=rng)
        blanket_assertions(perturb=inst.perturb, image=image, expected=expected)

        # Test callable
        blanket_assertions(
            perturb=GaussianPerturber(mean=mean, var=var, rng=rng),
            image=image,
            expected=expected
        )

    @pytest.mark.parametrize("image, expectation", [
        (np.zeros((256, 256, 3), dtype=np.uint8), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.uint8) * 255, does_not_raise()),
        (np.zeros((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.half), does_not_raise()),
        (np.ones((256, 356, 3), dtype=np.csingle),
            pytest.raises(NotImplementedError, match=r"Perturb not implemented for"))
    ])
    def test_no_perturbation(self, image: np.ndarray, expectation: ContextManager) -> None:
        """
        Run a dummy image through the perturber with settings for no
        perturbations, expect to get same image back (quantization errors
        possible). This attempts to isolate perturber implementation code
        from external calls to the extent that is possible.
        """
        with expectation:
            blanket_assertions(perturb=GaussianPerturber(mean=0, var=0), image=image, expected=image)

    @pytest.mark.parametrize("rng", [42, 12345])
    def test_rng(self, rng: int) -> None:
        """
        Ensure results are reproducible.
        """
        rng_assertions(perturber=GaussianPerturber, rng=rng)

    @pytest.mark.parametrize("rng, mean, var", [
        (42, 0.8, 0.25),
        (np.random.default_rng(12345), 0.3, 0.2)
    ])
    def test_configuration(self, rng: Union[np.random.Generator, int], mean: float, var: float) -> None:
        """
        Test configuration stability
        """
        inst = GaussianPerturber(rng=rng, mean=mean, var=var)
        for i in configuration_test_helper(inst):
            assert i.rng == rng
            assert i.mean == mean
            assert i.var == var

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"var": 0.75}, does_not_raise()),
        ({"var": 0}, does_not_raise()),
        ({"var": -10}, pytest.raises(ValueError, match=r"GaussianPerturber invalid var"))
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            GaussianPerturber(**kwargs)


class TestSpecklePerturber:
    @pytest.mark.parametrize("image, rng, mean, var, expected", [
        (np.ones((3, 3), dtype=np.uint8) * 255, 42, 0, 0.05,
            np.array([[255, 196, 255], [255, 144, 181], [255, 237, 254]]))
    ])
    def test_consistency(
        self,
        image: np.ndarray,
        rng: Union[np.random.Generator, int],
        mean: float,
        var: float,
        expected: np.ndarray
    ) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        # Test perturb interface directly
        inst = SpecklePerturber(mean=mean, var=var, rng=rng)
        blanket_assertions(perturb=inst.perturb, image=image, expected=expected)

        # Test callable
        blanket_assertions(
            perturb=SpecklePerturber(mean=mean, var=var, rng=rng),
            image=image,
            expected=expected
        )

    @pytest.mark.parametrize("image, expectation", [
        (np.zeros((256, 256, 3), dtype=np.uint8), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.uint8) * 255, does_not_raise()),
        (np.zeros((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
        (np.ones((256, 256, 3), dtype=np.half), does_not_raise()),
        (np.ones((256, 356, 3), dtype=np.csingle),
            pytest.raises(NotImplementedError, match=r"Perturb not implemented for"))
    ])
    def test_no_perturbation(self, image: np.ndarray, expectation: ContextManager) -> None:
        """
        Run a dummy image through the perturber with settings for no
        perturbations, expect to get same image back (quantization errors
        possible). This attempts to isolate perturber implementation code
        from external calls to the extent that is possible.
        """
        with expectation:
            blanket_assertions(perturb=SpecklePerturber(mean=0, var=0), image=image, expected=image)

    @pytest.mark.parametrize("rng", [42, 12345])
    def test_rng(self, rng: int) -> None:
        """
        Ensure results are reproducible.
        """
        rng_assertions(perturber=SpecklePerturber, rng=rng)

    @pytest.mark.parametrize("rng, mean, var", [
        (42, 0.8, 0.25),
        (np.random.default_rng(12345), 0.3, 0.2)
    ])
    def test_configuration(self, rng: Union[np.random.Generator, int], mean: float, var: float) -> None:
        """
        Test configuration stability
        """
        inst = SpecklePerturber(rng=rng, mean=mean, var=var)
        for i in configuration_test_helper(inst):
            assert i.rng == rng
            assert i.mean == mean
            assert i.var == var

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"var": 0.123}, does_not_raise()),
        ({"var": 0}, does_not_raise()),
        ({"var": -10}, pytest.raises(ValueError, match=r"SpecklePerturber invalid var"))
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            SpecklePerturber(**kwargs)
