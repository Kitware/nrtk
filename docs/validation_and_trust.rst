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

**Status as of February 2026.** Updates occur quarterly.

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
     - Aerial, maritime, overhead/WAMI, automotive, biometric (long-range)
   * - Modalities Coverage
     - ⚙️ In Progress
     - Still imagery → FMV (NRTK v1.1+); long-range video in progress
   * - Real-World Benchmarking
     - ⚙️ In Progress
     - RarePlanes, BDD100k, non-public WAMI and maritime datasets
   * - Independent Validation
     - ⚙️ In Progress
     - NAML'26 and MSS'26 (March 2026); SPIE'26 (April 2026)

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
* **MSS Parallel'26**: pyBSM-based perturbers evaluated on overhead imagery in WAMI format (non-public)
  and RarePlanes (public)
* **NAML'26**: Custom synthetic waterdroplet-on-lens perturbation on maritime/aerial data (non-public)
* **SPIE'26** (in progress): *Improving AI Test and Evaluation via Semantic Gap Detection and Generative
  Augmentation* — generative AI perturbation approaches on
  `BDD100k`_
* **Biometric application** (upcoming): Detection of individuals in long-range video; comparative analysis
  of pyBSM-based ground range simulation against real-world ground range
* Community feedback on perturbation realism is limited but growing

We track these in our `GitHub Issues <https://github.com/Kitware/nrtk/issues>`_ and prioritize based on
community feedback and DoD use-case requirements.

Validation Roadmap
==================

Embedding-space validation evaluates whether perturbations produce monotonic, stable, and interpretable
changes in model representations.

Initiated Nov'25 (Ongoing)
--------------------------

* ⚙️ Quantify perturbation effects in embedding space for photometric, geometric, and optical modules using
  standard baseline models

Planned for Mar'26
------------------

* 📋 Compare optical-perturbation outputs against real degraded imagery with known atmospheric and sensor
  parameters — detection of individuals in long-range video with comparative analysis of pyBSM ground
  range vs real-world ground range

Q1'26 (Dissemination & Reporting)
---------------------------------

* ⚙️ NAML'26 and MSS Parallel'26 conference presentations (March 2026)
* ⚙️ *Improving AI Test and Evaluation via Semantic Gap Detection and Generative Augmentation*
  — generative AI perturbation approaches on BDD100k (SPIE Defense + Security, April 26–30)

How You Can Help
================

Have real-world degraded imagery?
---------------------------------

If you can share operational data with known degradation factors (sensor specs, atmospheric conditions,
etc.), contact us at nrtk@kitware.com. This information directly improves our validation evidence.

Found unexpected behavior?
--------------------------

Report it in `GitHub Issues <https://github.com/Kitware/nrtk/issues>`_ with details about your use case.
User feedback is a critical validation input.

Using NRTK in your T&E workflow?
--------------------------------

Share your experience. Case studies help us understand what validation evidence matters most to the community.

Bottom Line
===========

NRTK accelerates the early stages of robustness evaluation by providing systematic, parametric perturbations.
It is not intended to replace operational testing, but to help users identify where deeper evaluation is
required. Validation evidence grows continuously, and this page is updated quarterly to reflect new findings.

**Questions?** nrtk@kitware.com | **Last Updated:** Feb. 26 2026

Related Resources
=================

* :doc:`/explanations/risk_factors` - Which perturbations map to which operational risks
* :doc:`/tutorials/testing_and_evaluation_notebooks` - Examples of NRTK in realistic testing workflows
* :doc:`/release_notes/index` - Validation updates and known issue resolutions

Publications & Presentations
=============================

.. note::
   Entries will be updated with full citations after proceedings are released.

`Naval Applications of Machine Learning (NAML'26) <https://naml2026.org/>`_ — March 2–5, 2026
   *Establishing Trust in Maritime Detection Models with the Natural Robustness Toolkit*
   — Custom synthetic waterdroplet-on-lens perturbation; maritime/aerial domain (non-public data)

`Military Sensing Symposia (MSS Parallel'26)`_ — March 2–6, 2026
   *Understanding Sensor-based Robustness of Object Detection Models
   for Overhead Imagery*
   — pyBSM-based perturbers; WAMI (non-public) and RarePlanes (public)

`SPIE Defense + Security (SPIE'26)`_ — April 26–30, 2026 *(in preparation)*
   *Improving AI Test and Evaluation via Semantic Gap Detection
   and Generative Augmentation*
   — Generative AI perturbation approaches on `BDD100k`_

.. _Military Sensing Symposia (MSS Parallel'26):
   https://mssconferences.org/public/meetings/conferenceDetail.aspx?enc=GCQKXtAOLmd8QhTdmLLt9Q%3D%3D
.. _SPIE Defense + Security (SPIE'26):
   https://spie.org/conferences-and-exhibitions/defense-and-security
.. _BDD100k:
   https://bair.berkeley.edu/blog/2018/05/30/bdd/

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
