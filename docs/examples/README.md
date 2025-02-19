# Examples
This directory hosts the NRTK examples.

### Implementation Guide
Example notebooks should have an "Open in Colab" button.
See other example notebooks for information on how to add a Colab cell.

Example notebooks that download or generate data should place that data in the `data` sub-directory.

### Requirements
Most of the examples require [Jupyter Notebook](https://jupyter.org/) and
[PyTorch](https://pytorch.org/).

Install these manually using `pip` or with the following command:

```bash
poetry install -E example_deps
```

Some notebooks may require additional dependencies.
See the first cell of each notebook ("Set Up the Environment") for instructions on how to install
the relevant packages.

### Run the Notebooks from Colab
Most of the notebooks have an "Open in Colab" button.
Right-click on the button and select "Open Link in New Tab" to start a Colab
page with the corresponding notebook content.

To use GPU resources through Colab, remember to change the runtime type to
`GPU`:

1. From the `Runtime` menu select `Change runtime type`.
1. Choose `GPU` from the drop-down menu.
1. Click `SAVE`.

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

### Data

Some notebooks may require additional data. This data will be downloaded when
running the notebook.

### Encountering Issues

For issues relating to NRTK functionality or running an example, 
create an issue on the [repository](https://github.com/Kitware/nrtk/issues).

---

This README is adapted from [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
