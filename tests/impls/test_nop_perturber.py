import numpy as np

from nrtk.impls.perturb_image.nop_perturber import NOPPerturber


def test_perturb() -> None:
    """
    Run on a dummy image to ensure output matches expectations.
    Test using the perturb implementation directly.
    """
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    inst = NOPPerturber()
    out_image = inst.perturb(dummy_image)

    assert np.array_equal(dummy_image, out_image)


def test_callable() -> None:
    """
    Run on a dummy image to ensure output matches expectations.
    Test using the ``__call__`` alias.
    """
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    inst = NOPPerturber()
    out_image = inst(dummy_image)

    assert np.array_equal(dummy_image, out_image)


def test_side_effects() -> None:
    """
    Ensure implementation imparts no side effects to the input.
    """
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_image_copy = np.copy(dummy_image)
    inst = NOPPerturber()
    _ = inst(dummy_image)

    assert np.array_equal(dummy_image, dummy_image_copy)
    assert not np.shares_memory(dummy_image, dummy_image_copy)


def test_config() -> None:
    inst = NOPPerturber()
    config = inst.get_config()

    # Config should be empty, but not None
    assert config is not None
    assert not config
