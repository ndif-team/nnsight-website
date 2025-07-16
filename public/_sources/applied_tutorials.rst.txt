:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', (event) => {
      document.querySelectorAll('h5.card-title').forEach(el => {
      el.style.margin = '0';
      });
   });
   </script>

   <style>
      .toctree-wrapper {
         display: none !important;
      }
      h5 {
         margin-top: 0 !important;
      }
   </style>


Mini Paper Tutorials
=====================

Many researchers in interpretability write their experiments in :bdg-primary:`nnsight`, and you can too! Follow the applied tutorials below to recreate and experiment with results from recent published papers.

.. grid:: 1 1 2 2
   :class-container: tutorial-card-section
   :gutter: 3

   .. grid-item-card:: 
      :link: notebooks/mini-papers/csordas_llm_depth.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-network-wired fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Do LMs Use Their Depth?</h5>
               <p class="card-text">Skip layers to see how deeply LMs think</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/mini-papers/marks_geometry_of_truth.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-shapes fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Geometry of Truth</h5>
               <p class="card-text">Steer what models think is or isn't true</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/mini-papers/todd_function_vectors.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-arrows-turn-to-dots fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Function Vectors</h5>
               <p class="card-text">Steer models between tasks</p>
            </div>
         </div>

Submit Your Own Ideas!
----------------------
We welcome contributions to our list of applied examples. If you have an example of using :bdg-primary:`nnsight` in your own project, or you have an idea for a tutorial that would be helpful to you, please share it with us!

.. grid:: 1 1 2 2
   :padding: 2

   .. grid-item::
      .. div:: sd-text-center

         .. button-link:: https://forms.gle/T8aLr1YLbBAqu8Ra9
            :color: primary
            :shadow:

            Submit Tutorial

   .. grid-item::
      .. div:: sd-text-center
         
         .. button-link:: https://forms.gle/KfwrtbW2j9u8MoHL8
            :color: primary
            :shadow:

            Submit Suggestion


.. toctree::
   :maxdepth: 1

   notebooks/mini-papers/csordas_llm_depth
   notebooks/mini-papers/marks_geometry_of_truth
   notebooks/mini-papers/todd_function_vectors