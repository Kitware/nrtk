# Examples

This directory hosts the NRTK examples.

## Implementation Guide

Example notebooks should have an "Open in Colab" button. See other example
notebooks for information on how to add a Colab cell.

Example notebooks that download or generate data should place that data in the
`data` sub-directory.

Notebooks should also be included in the CI jobs so their execution is checked
automatically. Document any exceptions in the section below.

### Notebook CI Exclusions

For each excluded notebook, record its repository-relative path and the
rationale for exclusion.

<!-- Add an entry for each excluded notebook:
- `docs/examples/path/to/notebook.ipynb`: Rationale for exclusion.
-->

## Requirements

To run a notebook locally, install [Jupyter Notebook](https://jupyter.org/):

```bash
python -m pip install notebook
```

Run the notebook's setup cells before the rest of the example. These cells
install its Python dependencies, including PyTorch when needed.

## Download and Run Locally

On a notebook's documentation page, select **Jupyter notebook** at the top of
the page to save the `.ipynb` file. Open the downloaded file with Jupyter:

```bash
jupyter notebook path/to/notebook.ipynb
```

Read the notebook's setup and data instructions before running the example.

### Notebooks with Repository Helper Files

Some notebooks currently need helper files from this repository in addition to
the `.ipynb` file:

- [Framewise Video](maite/framewise_video.ipynb) uses
  `docs/examples/notebook_utils/`.
- The [saliency workflows](xaitk_saliency_workflow/README.md) use
  `docs/examples/xaitk_saliency_workflow/utils/`.

For these notebooks, download or clone the repository and keep its directory
structure intact. Open the notebook with Jupyter and run its setup cells as
above. For contributing to NRTK, follow the
[source installation instructions](../getting_started/installation.rst).

## Additional Installs & Extras

Many notebooks depend on optional extras from `nrtk`, which must be installed
for the notebook to run correctly. These installs are included at the top of
each notebook.

To support both local development and Colab execution (see next section), we use
the following installation pattern:

```bash
!{sys.executable} -c "import nrtk" || !{sys.executable} -m pip install -q "nrtk[maite,albumentations]>=X.Y.Z"
```

Here, `X.Y.Z` is the minimum `nrtk` version required for the notebook.

This may look unusual, but it solves a practical issue:

- On Colab, the package must be installed from PyPI rather than the local
  development environment.
- During development, we want to use the local version of `nrtk`.
- Without this approach, notebooks would fail prior to a release, since the
  required version wouldn’t yet be available on PyPI.

This check-and-install pattern ensures notebooks work in both contexts.

To help maintain a reproducible environment and simplify debugging (especially
with `nrtk`’s extras), we also provide a utility function:

<!-- pytestmark: skip -->

```python
print_extras_status()
```

This prints out all extras, their dependencies, and whether they are installed.

## Run the Notebooks from Colab

Most of the notebooks have an "Open in Colab" button. Right-click on the button
and select "Open Link in New Tab" to start a Colab page with the corresponding
notebook content.

To use GPU resources through Colab, remember to change the runtime type to
`GPU`:

1. From the `Runtime` menu select `Change runtime type`.
2. Choose `GPU` from the drop-down menu.
3. Click `SAVE`.

This will reset the notebook and may ask you if you are a robot (these
instructions assume you are not).

Running:

```bash
!nvidia-smi
```

in a cell will verify this has worked and show you what kind of hardware you
have access to.

**Note that after setting up the environment, you may need to "Restart**
**Runtime" to resolve package version conflicts.**

## Data

Some notebooks may require additional data. Most of this data will be downloaded
when running the notebook. However, some notebooks rely on
[Git LFS](https://git-lfs.com/) being installed for data access.

## Encountering Issues

For issues relating to NRTK functionality or running an example, create an issue
on the [repository](https://github.com/Kitware/nrtk/issues).

______________________________________________________________________

This README is adapted from
[MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
