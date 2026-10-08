Best Practices for Applying Image Perturbations to FMV Data
===========================================================

The :class:`~nrtk.impls.perturb_video.FramewisePerturber` applies a specified NRTK image perturber independently to each
frame of a given FMV stream. This provides a convenient way to reuse existing image perturbations on video, without
requiring a dedicated video perturbation.

However, the Framewise Perturber is **not a general-purpose video perturbation mechanism**. Applying an image
perturbation independently to each frame does not guarantee temporal or physical consistency between frames.

For effects that evolve over time or depend on scene motion, prefer a video-aware perturbation when one is available,
as NRTK video perturbations can incorporate temporal context where necessary. For example,
:class:`~nrtk.impls.perturb_video.optical.TurbulenceVideoPerturber` models temporally evolving blur and jitter, whereas
independently applying :class:`~nrtk.impls.perturb_image.optical.otf.TurbulenceAperturePerturber` to successive frames
does not.

.. important::
  When a video-aware perturbation exists, prefer it over applying the corresponding image perturbation independently to
  each frame.

Understanding ``is_static``
---------------------------

Some NRTK perturbers with random state provide an ``is_static`` parameter. When used with the Framewise Perturber, this
parameter controls whether the perturber's random state is reset between frames.

.. important::
  Repeating the same perturbation across frames does **not** guarantee a temporally realistic video.

In particular, ``is_static=True`` does **not** mean that adjacent frames guaranteed to be physically consistent or
plausible.

Choosing a Perturbation
-----------------------

The suitabilty of an image perturbation for the Framewise Perturber depends on the real-world effect being
modeled.

Before using an image perturbation with FMV data, consider:

1. What risk factor are you trying to simulate?
2. Can the effect reasonably be modeled independently on each frame?
3. Does the effect need to evolve over time or remain consistent with scene motion?
4. Is a video-aware perturbation available that better represents the effect?

If the effect can reasonably be applied independently to each FMV frame, the Framewise Perturber may be appropriate.

Good Candidates
^^^^^^^^^^^^^^^

Image perturbations are generally good candidates when their effects can reasonably be applied independently to
individual frames. Examples include:

* :class:`~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber`
* :class:`~nrtk.impls.perturb_image.photometric.enhance.ContrastPerturber`
* :class:`~nrtk.impls.perturb_image.photometric.enhance.ColorPerturber`
* :class:`~nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber`
* :class:`~nrtk.impls.perturb_image.photometric.blur.GaussianBlurPerturber`
* :class:`~nrtk.impls.perturb_image.photometric.blur.MedianBlurPerturber`
* :class:`~nrtk.impls.perturb_image.photometric.blur.GaussianBlurPerturber`
* :class:`~nrtk.impls.perturb_image.geometric.random.RandomCropPerturber` when ``is_static=True``
* :class:`~nrtk.impls.perturb_image.geometric.random.RandomRotationPerturber` when ``is_static=True``
* :class:`~nrtk.impls.perturb_image.geometric.random.RandomScalePerturber` when ``is_static=True``
* :class:`~nrtk.impls.perturb_image.geometric.random.RandomTranslationPerturber` when ``is_static=True``

Note, these are examples rather than universal recommendations. The appropriateness of a perturber depends on the risk
factor being simulated and the degree of temporal consistency required.

Poor Candidates
^^^^^^^^^^^^^^^

Perturbations that model time-varying physical phenomena or transformations that depend on scene content are generally
poor candidates for independent frame-by-frame application. Examples include:

* :class:`~nrtk.impls.perturb_image.environment.WaterDropletPerturber`

  * Note this perturber may be reasonable in limited scenarios, such as a short stream where gravitational effects
    are negligible and ``is_static=True``.

* :class:`~nrtk.impls.perturb_image.generative.DiffusionPerturber`
* Perturbers representing rapidly changing optical effects, such as jitter or turbulence.

For these effects, independently perturbing each frame can introduce unrealistic frame-to-frame changes, even when the
same perturbation and parameters are used throughout the stream.
