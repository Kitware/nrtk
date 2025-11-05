* FIX: Fixed ``OneStepPerturbImageFactory`` to preserve float values by passing ``to_int=False``
  to parent ``StepPerturbImageFactory``. Previously, float values like ``theta_value=0.5``
  were incorrectly converted to ``0`` due to the parent's default ``to_int=True`` behavior.

* CHANGE: Changed the default value of the ``to_int`` parameter in ``StepPerturbImageFactory``
  from ``True`` to ``False``. This makes float values the default behavior. Users who require
  integer values should now explicitly pass ``to_int=True`` when instantiating the factory.
