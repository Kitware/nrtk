import numpy as np

from nrtk.impls.perturb_image.photometric._noise.noise_perturber_mixin import NoisePerturberMixin

test_rng = np.random.default_rng()


def rng_assertions(perturber: type[NoisePerturberMixin], rng: int) -> None:
    """Test that output is reproducible if a rng or seed is provided.

    :param perturber: SKImage random_noise perturber class of interest.
    :param rng: Seed value.
    """
    dummy_image_a = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
    dummy_image_b = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

    # Test as seed value
    inst_1 = perturber(rng=rng)
    out_1a, _ = inst_1(image=dummy_image_a)
    out_1b, _ = inst_1(image=dummy_image_b)
    inst_2 = perturber(rng=rng)
    out_2a, _ = inst_2(image=dummy_image_a)
    out_2b, _ = inst_2(image=dummy_image_b)
    assert np.array_equal(out_1a, out_2a)
    assert np.array_equal(out_1b, out_2b)

    # Test generator
    inst_3 = perturber(rng=np.random.default_rng(rng))
    out_3a, _ = inst_3(image=dummy_image_a)
    out_3b, _ = inst_3(image=dummy_image_b)
    inst_4 = perturber(rng=np.random.default_rng(rng))
    out_4a, _ = inst_4(image=dummy_image_a)
    out_4b, _ = inst_4(image=dummy_image_b)
    assert np.array_equal(out_3a, out_4a)
    assert np.array_equal(out_3b, out_4b)
