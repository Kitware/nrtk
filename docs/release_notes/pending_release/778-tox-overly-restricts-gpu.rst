* Fixed notebook CI jobs using a shared runner tag, which prevented
  GPU-dependent notebooks from being tested on CUDA. Each notebook now
  specifies its own runner tag so that GPU notebooks run on GPU runners.

* Fixed ``tox`` blanket ``CUDA_VISIBLE_DEVICES`` override hiding GPUs from
  the ``papermill`` environment. The ``papermill`` environment now passes
  through the host's ``CUDA_VISIBLE_DEVICES`` so that GPU-dependent
  notebooks can access CUDA when run on a GPU runner.
