* Fixed ``_get_device`` CUDA availability check to support specific GPU
  device strings (e.g., ``"cuda:0"``, ``"cuda:1"``). Previously only the
  bare ``"cuda"`` string was matched, so requests for a specific GPU would
  bypass the fallback-to-CPU logic when CUDA was unavailable.
