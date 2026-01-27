================
Containerization
================

AUKUS Container
===============

To support users that require tools for an ML T&E workflow, we define a container that would
accept an input dataset and apply perturbations to the entire dataset. The perturbed images
should be saved to disk and then
the container will shut down. In order to support this workflow, the AUKUS container was
created.

Given a COCO dataset and an NRTK factory configuration file, the AUKUS container is able to
generate perturbed images for each image in the dataset. Each perturbed image will be saved
to a given output directory as an individual image file. Once all perturbed images are saved,
the container will terminate.

How to Use
----------
To run the AUKUS container, use the following command:
``docker run -v /path/to/input:/input/:ro -v /path/to/output:/output/ nrtk-perturber``
This will mount the inputs to the correct locations and use the ``nrtk-perturber`` CLI script
with the default args. The CLI script will attempt to load a COCO dataset from the ``/input/data/dataset/``
directory, save perturbed images to
``/output/data/result/``, and load a config file named ``nrtk_config.json``. The ``dataset`` directory
and ``nrtk_config.json`` file must be in the directory mounted to ``/input/``.

Input Arguments
---------------

The container accepts three input arguments:

   * ``dataset_dir``: input COCO dataset
   * ``output_dir``: directory to store the generated saliency maps
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

Command line options take precendence over environment variables if both are provided.

Note: The values for ``dataset_dir`` and ``config_file`` should be written from the
perspective of the container (i.e. ``/path/on/container/dataset_dir/`` instead of
``/path/on/local/machine/dataset_dir/``)

Limitations
-----------

Currently, the AUKUS container supports the loading of only COCO datasets.
Any existing dataset must be converted to a COCO dataset before using the AUKUS container. Please
see `KWCOCO documentation <https://kwcoco.readthedocs.io/en/main/>`_ for more information on COCO
datasets.
