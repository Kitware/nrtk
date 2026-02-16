====================
Validation and Trust
====================

This page summarizes how NRTK is validated today, the available evidence, the remaining gaps, and how users
should interpret perturbation-based robustness results (see :doc:`/explanations/nrtk_explanation`). It is not a full T&E
manual, but a transparency resource for anyone integrating NRTK into evaluation workflows.

NRTK provides rapid, cost-effective perturbation testing to identify potential model vulnerabilities and
robustness gaps. The perturbations are designed to be indicative rather than authoritative. They provide
fast, low-cost stress tests to expose potential vulnerabilities, not statistically definitive operational
predictions.

.. important::
   NRTK perturbations are designed to complement, not replace, complete model validation. They are one tool
   in a comprehensive T&E strategy, not a replacement for evaluation with real operational data.

Validation Status
=================

We're transparent about what's verified, what's in progress, and what's planned.

**Status as of November 2025.** Updates occur quarterly.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Validation Aspect
     - Status
     - Details
   * - Algorithmic Correctness
     - ✅ Verified
     - Unit and integration testing; continuous integration
   * - Reproducibility
     - ✅ Verified
     - Deterministic outputs with fixed seeds; documented test cases
   * - Parameter Validation
     - ✅ Verified
     - Range checks, unit consistency, fail-fast logic, and default-parameter justification
   * - Cross-Tool Integration
     - ✅ Verified
     - MAITE compliance; tested with DataEval, XAITK
   * - Operational Realism
     - ⚙️ In Progress
     - Collecting real-world degraded imagery for comparison
   * - Domain Coverage
     - ⚙️ In Progress
     - Expanding from aerial to ground/surface domains
   * - Modalities Coverage
     - ⚙️ In Progress
     - Expanding from still imagery to full-motion video
   * - Real-World Benchmarking
     - ⚙️ In Progress
     - Comparison studies with operational datasets
   * - Independent Validation
     - 📋 Planned
     - External research partnerships; peer review

How we validate:

* **Algorithmic**: Mathematical correctness of perturbation implementations
* **Empirical**: Comparison with real-world degraded imagery where available
* **Operational**: Feedback from T&E engineers using NRTK in actual workflows
* **Methodological**: Experimentally validated using methodology grounded in academic literature
* **Reproducibility**: Consistent outputs across platforms and versions

.. note::
   For module-specific validation details, see:

   * :doc:`/reference/api/implementations` - Individual perturbation modules with implementation details
   * :doc:`/explanations/risk_factors` - Mapping between operational risks and NRTK perturbations

   Each perturbation module page includes parameter documentation and usage examples.

When to Use NRTK
================

✅ Good For
-----------

* Early-stage robustness screening
* Parameter sensitivity analysis
* Identifying potential failure modes
* Data augmentation during training
* Comparing robustness across models
* Cost-performance trade-off studies

⚠️ Supplement with Mission-Representative Data
-----------------------------------------------

NRTK is reliable for perturbation-driven insights, but not a substitute for mission-representative data.
Combine NRTK results with operational evaluation for:

* Final deployment decisions
* Safety-critical systems
* Novel operational environments

❌ Not Appropriate For
----------------------

* Sole source of model validation
* Regulatory certification or compliance
* Precise predictions of real-world performance

Known Limitations
=================

We document limitations openly to help users make informed decisions:

Current Scope
-------------

* Optimized for static images (FMV support in development)
* Primary focus on classification and detection (segmentation/tracking in development)
* Examples emphasize aerial imaging (expanding to ground/surface domains)

Technical Constraints
---------------------

* **Spectral domain assumptions**: Defaults assume visible-spectrum RGB imagery. IR/SAR/HSI sensors require
  domain-appropriate optical parameters; NRTK does not provide full spectral physics for all
  modalities.
* **Perturbation composition effects**: Applying perturbations sequentially may not perfectly replicate
  real-world conditions where effects occur simultaneously. For example, sensor noise and atmospheric blur
  interact differently than applying blur then noise in post-processing.

Validation Evidence
-------------------

* Real-world imagery comparison ongoing; results published as available (e.g. ReadTheDocs, GitHub, and
  academic publications)
* Community feedback on perturbation realism is limited but growing

We track these in our GitHub Issues and prioritize based on community feedback and DoD use-case requirements.

Validation Roadmap
==================

Embedding-space validation evaluates whether perturbations produce monotonic, stable, and interpretable
changes in model representations.

Nov'25 (Current)
-------------------------

* ⚙️ Quantify perturbation effects in embedding space for photometric, geometric, and optical modules using
  standard baseline models

Dec'25 (Future)
---------------

* 📋 Compare optical-perturbation outputs against real degraded imagery with known atmospheric and sensor
  parameters

Early Q1'26 (Future)
--------------------

* 📋 Release reproducible validation benchmarks demonstrating monotonicity, sensitivity, and cross-model
  consistency for all perturbation categories

How You Can Help
================

Have real-world degraded imagery?
---------------------------------

If you can share operational data with known degradation factors (sensor specs, atmospheric conditions,
etc.), contact us at nrtk@kitware.com. This information directly improves our validation evidence.

Found unexpected behavior?
--------------------------

Report it in GitHub Issues with details about your use case. User feedback is a critical validation input.

Using NRTK in your T&E workflow?
--------------------------------

Share your experience. Case studies help us understand what validation evidence matters most to the community.

Bottom Line
===========

NRTK accelerates the early stages of robustness evaluation by providing systematic, parametric perturbations.
It is not intended to replace operational testing, but to help users identify where deeper evaluation is
required. Validation evidence grows continuously, and this page is updated quarterly to reflect new findings.

**Questions?** nrtk@kitware.com | **Last Updated:** Nov. 21 2025

Related Resources
=================

* **Operational Risk Factors** - Which perturbations map to which operational risks
* **Parameter Defaults Documentation** - Why specific defaults were chosen
* **T&E Guides** - Examples of NRTK in realistic testing workflows
* **Release Notes** - Validation updates and known issue resolutions

How to Cite
===========

When referencing NRTK validation in reports, briefings, or evaluation documentation:

**Recommended citation:**

   Kitware, Inc. (2025). *NRTK Validation & Trust Documentation*. Natural Robustness Toolkit.
   Retrieved from https://nrtk.readthedocs.io/en/stable/validation_and_trust.html

**BibTeX:**

.. code-block:: bibtex

   @misc{nrtk_validation_2025,
     title        = {NRTK Validation \& Trust Documentation},
     author       = {{Kitware, Inc.}},
     year         = {2025},
     howpublished = {\url{https://nrtk.readthedocs.io/en/stable/validation_and_trust.html}},
     note         = {Accessed: [Insert Date]}
   }

References
==========

Related Tools and Standards
----------------------------

* **MAITE (MIT-LL AI Technology Evaluation)**: Standardized protocols for AI T&E workflows.
  https://mit-ll-ai-technology.github.io/maite/
* **DataEval**: JATIC tool for dataset quality and coverage analysis.
  https://dataeval.readthedocs.io/
* **XAITK-Saliency**: Explainability toolkit for understanding model decisions.
  https://xaitk-saliency.readthedocs.io/
