* Added a check to ``TurbulenceAperturePerturber._create_simulator`` to ensure ``slant_range``
  is greater than or equal to ``altitude``.

* Updated ``TestTurbulenceAperturePerturber.test_configuration_bounds`` to test for slant range less than altitude.
