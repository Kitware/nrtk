================
Containerization
================

``nrtk-perturber`` Container
============================

To support users that require tools for an ML T&E workflow, we define a container that would
accept an input dataset and apply perturbations to the entire dataset. The perturbed images
should be saved to disk and then
the container will shut down. In order to support this workflow, the ``nrtk-perturber`` container was
created.

Given a COCO dataset and an NRTK factory configuration file, the ``nrtk-perturber`` container is able to
generate perturbed images for each image in the dataset. Each perturbed image will be saved
to a given output directory as an individual image file. Once all perturbed images are saved,
the container will terminate.

Image Tags
----------

Container images are published to Harbor at
``harbor.jatic.net:443/kitware/nrtk/nrtk-perturber``.

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
with the default args. The CLI script will attempt to load a COCO dataset from the ``/input/data/dataset/``
directory, save perturbed images to
``/output/data/result/``, and load a config file named ``nrtk_config.json``. The ``dataset`` directory
and ``nrtk_config.json`` file must be in the directory mounted to ``/input/``.

.. note::

   Ensure the ``output`` directory is writable by non-root users.

Input Arguments
---------------

The container accepts three input arguments:

   * ``dataset_dir``: input COCO dataset
   * ``output_dir``: directory to store the generated perturbed images
   * ``config_file``: configuration file specifying the ``PerturbImageFactory`` params for image perturbation

These can be controlled in two ways: ``Environment Variables`` or ``CLI Options``.

The following environment variables are used by default:

   * ``INPUT_DATASET_PATH``: Path to input dataset (default: ``/input/data/dataset/``)
   * ``OUTPUT_DATASET_PATH``: Path to output directory (default: ``/output/data/result/``)
   * ``CONFIG_FILE``: Path to config file (default: ``/input/nrtk_config.json``)

To override defaults, use the ``-e`` flag:
``docker run -e INPUT_DATASET_PATH=/custom/path ... nrtk-perturber``

If a user does not want to use environment variables, they can use command line options. After the container name,
the user can use the following flags:

   * ``--dataset_dir`` or ``-d``: Path to input dataset
   * ``--output_dir`` or ``-o``: Path to output directory
   * ``--config_file`` or ``-c``: Path to config file

Command line options take precedence over environment variables if both are provided.

.. note::

   The values for ``dataset_dir`` and ``config_file`` should be written from the
   perspective of the container (i.e. ``/path/on/container/dataset_dir/`` instead of
   ``/path/on/local/machine/dataset_dir/``)

Error Codes
-----------

``101``: Could not identify an annotations file for COCO dataset.


Image Verification and SBOM
---------------------------

Container images published to Harbor are signed with `cosign <https://docs.sigstore.dev/cosign/overview/>`_
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

Limitations
-----------

Currently, the ``nrtk-perturber`` container supports loading only COCO datasets.
Any existing dataset must be converted to a COCO dataset before using the ``nrtk-perturber`` container. Please
see `KWCOCO documentation <https://kwcoco.readthedocs.io/en/main/>`_ for more information on COCO
datasets.
