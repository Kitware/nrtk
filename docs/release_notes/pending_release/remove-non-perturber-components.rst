* Removed scoring interfaces and implementations (``ScoreDetections``, ``ScoreClassifications``,
  ``COCOScorer``, ``NOPScorer``, ``RandomScorer``, ``ClassAgnosticPixelwiseIoUScorer``).

* Removed image metric interfaces and implementations (``ImageMetric``, ``NIIRSImageMetric``,
  ``SNRImageMetric``).

* Removed blackbox response generator interfaces and implementations
  (``GenerateBlackboxResponse``, ``GenerateObjectDetectorBlackboxResponse``,
  ``GenerateClassifierBlackboxResponse``, ``SimpleGenericGenerator``, ``SimplePybsmGenerator``).

* Removed associated example notebooks (``coco_scorer.ipynb``, ``simple_generic_generator.ipynb``,
  ``simple_pybsm_generator.ipynb``, ``compute_image_metric.ipynb``).

* Updated documentation to reflect the removal of these components.
