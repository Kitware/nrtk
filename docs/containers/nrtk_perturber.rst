================
Containerization
================

``nrtk-perturber`` Container
============================

To support users that require tools for an ML T&E workflow, we define a container that would
accept an input dataset and apply perturbations to the entire dataset. The perturbed data
should be saved to disk and then the container will shut down.
In order to support this workflow, the ``nrtk-perturber`` container was created.

Given a ``datamaite`` dataset and an NRTK factory configuration file, the ``nrtk-perturber`` container is able to
generate perturbed data for each item in the dataset. Each perturbed item will be saved
to a given output directory. Once all perturbed data is saved, the container will terminate.

.. seealso::
   See :doc:`/development/system_requirements` for the container's supported
   architecture and recommended minimum hardware (CPU, GPU, memory, storage),
   including guidance for running under Kubernetes.

Image Tags
----------

Container images are published to Harbor at ``harbor.jatic.net:443/kitware/nrtk/nrtk-perturber``.

**Release tags** (installed from PyPI, version matches nrtk):

* ``X.Y.Z``, ``X.Y``, ``X``, ``latest``

**Development tags** (built from source):

* ``main`` -- always reflects the latest default branch
* ``<branch-slug>`` -- built for a specific branch (e.g. ``dev-42-container-update``), cleaned up after merge

How to Use
----------
To run the ``nrtk-perturber`` container, use the following command:
``docker run -v /path/to/input:/input/:ro -v /path/to/output:/output/ harbor.jatic.net:443/kitware/nrtk/nrtk-perturber:latest``
This will mount the inputs to the correct locations and use the CLI script
with the default args. The CLI script will attempt to load a dataset from the ``/input/data/dataset/``
directory, save perturbed data to
``/output/data/result/``, and load a config file named ``nrtk_config.json``. The ``dataset`` directory
and ``nrtk_config.json`` file must be in the directory mounted to ``/input/``.

.. note::

   Ensure the ``output`` directory is writable by non-root users.

Input Arguments
---------------

The container accepts eight input arguments:

* ``dataset_dir``: input dataset
* ``input_dataset_format``: input dataset format
* ``output_dir``: directory to store the generated perturbed data
* ``output_dataset_format``: output dataset format
* ``config_file``: configuration file specifying the :class:`~nrtk.interfaces.PerturbFactory` params for perturbation
* ``combine_output``: boolean to control if output should be combined to one dataset or not
* ``overwrite``: Boolean to control ``datamaite.write`` mode. If enabled, ``datamaite.write`` will use
  ``mode="replace"`` when writing dataset(s). This will delete all files/folders in the output directory.
  If disabled, ``datamaite.write`` will use ``mode="error"`` when writing dataset(s)
* ``enable_experimental``: boolean for enabling experimental features

These can be controlled in two ways: ``Environment Variables`` or ``CLI Options``.

The following environment variables are used by default:

* ``INPUT_DATASET_PATH``: Path to input dataset (default: ``/input/data/dataset/``)
* ``INPUT_DATASET_FORMAT``: Input dataset format (default: ``COCO``)
* ``OUTPUT_DATASET_PATH``: Path to output directory (default: ``/output/data/result/``)
* ``OUTPUT_DATASET_FORMAT``: Output dataset format (default: ``COCO``)
* ``CONFIG_FILE``: Path to config file (default: ``/input/nrtk_config.json``)
* ``COMBINE_OUTPUT``: boolean to control ``combine_output`` (default: ``false``)
* ``OVERWRITE``: boolean to control ``overwrite`` (default: ``false``)
* ``ENABLE_EXPERIMENTAL``: boolean to control ``enable_experimental``. (default: ``false``).

To override defaults, use the ``-e`` flag:
``docker run -e INPUT_DATASET_PATH=/custom/path ... nrtk-perturber``

If a user does not want to use environment variables, they can use command line options. After the container name,
the user can use the following flags:

* ``--dataset_dir`` or ``-d``: Path to input dataset
* ``--input_dataset_format`` or ``-i``: Input dataset format
* ``--output_dir`` or ``-o``: Path to output directory
* ``--output_dataset_format`` or ``-u``: Output dataset format
* ``--config_file`` or ``-c``: Path to config file
* ``--combine_output`` or ``-m``: Boolean for combining outputs
* ``--overwrite`` or ``-r``: Boolean for controlling write mode
* ``--enable_experimental`` or ``-e``: Boolean for enabling experimental features

Command line options take precedence over environment variables if both are provided.

.. note::

   The values for ``dataset_dir`` and ``config_file`` should be written from the
   perspective of the container (i.e. ``/path/on/container/dataset_dir/`` instead of
   ``/path/on/local/machine/dataset_dir/``)

Error Codes
-----------

``101``: Error occurred while loading input dataset.

``102``: Input dataset is empty.

``103``: Dataset task is not supported.

``104``: Attempting to use experimental features without setting ``enable_experimental`` to True

``105``: Invalid ``input_dataset_format``.

``106``: Invalid ``output_dataset_format``.

``107``: Output directory is not empty and ``--overwrite`` was not given

Notes
-----

All image datasets are written out as ``PNG`` formatted images.

If existing image metadata contains area and/or segmentation attributes, both attributes
will be set to ``None`` in the augmented dataset(s).

If a user wants to pass datum metadata (i.e. ``img_gsd`` for an image), a file named
``datum_metadata.json`` should exist in the ``dataset_dir``.


Image Verification and SBOM
---------------------------

Container images published to Harbor are signed with `cosign <https://docs.sigstore.dev/quickstart/quickstart-cosign/>`_
and include a signed SBOM (Software Bill of Materials) attestation in SPDX format.

To verify the image signature::

   cosign verify --key cosign.pub \
     harbor.jatic.net:443/kitware/nrtk/nrtk-perturber:<tag>

To verify and view the SBOM attestation::

   cosign verify-attestation --key cosign.pub --type spdx \
     harbor.jatic.net:443/kitware/nrtk/nrtk-perturber:<tag>

To download the SBOM for inspection::

   cosign download attestation \
     harbor.jatic.net:443/kitware/nrtk/nrtk-perturber:<tag> \
     | jq -r '.payload' | base64 -d | jq .

The ``cosign.pub`` public key is located in the repository root.

.. note::

   The SBOM is also available as a CI pipeline artifact (``sbom.spdx.json``) on the
   build job that produced the image. The vulnerability scan report
   (``gl-container-scanning-report.json``) is available there as well.
