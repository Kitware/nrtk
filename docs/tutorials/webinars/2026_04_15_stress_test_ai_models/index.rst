How to Stress-Test Your AI Models Without Collecting New Data
==============================================================

**April 15, 2026 — Brandon RichardWebster, Ph.D. & Emily Veenhuis, Senior R&D Engineers, Kitware**

AI models rarely fail in controlled lab environments — they fail in the real world. Field data is the gold standard
for evaluating robustness, but it's expensive to collect across every condition that matters. This webinar shows how
**synthetic perturbation testing** with NRTK complements field data — helping you identify where the model is most
fragile, where to focus development efforts, and where to add mitigation strategies.

In the accompanying notebook, we:

1. **Install** the NRTK package.
2. **Show perturbations** on sample imagery to simulate real-world conditions.
3. **Set up T&E analysis** through MAITE.
4. **Run controlled parameter sweeps** to stress-test model performance.
5. **Perform a light robustness evaluation**.

----

.. grid:: 1
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: Hands-on Webinar Notebook
      :class-card: sd-border-2
      :columns: 12

      Walk through the full end-to-end workflow presented in the webinar: image perturbation, MAITE-driven
      evaluation, and parameter sweeps — applied to an aerial object-detection scenario.

      +++

      .. button-ref:: /tutorials/webinars/2026_04_15_stress_test_ai_models/notebook
         :color: primary
         :outline:

         Open the Notebook →

.. toctree::
   :hidden:

   notebook
