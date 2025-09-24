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
      .bd-sidebar h2.sidebar-title {
         font-weight: bold !important;
      }
      .get-started {
         background-color: var(--tutorial-category-get-started) !important;
      }
      .probing {
         background-color: var(--tutorial-category-probing) !important;
      }
      .steering {
         background-color: var(--tutorial-category-steering) !important;
      }
      .causal {
         background-color: var(--tutorial-category-causal) !important;
      }
   </style>


Main Tutorials
=====================

.. grid:: 2 2 4 4
   :class-container: tutorial-card-section
   :gutter: 3

   .. grid-item-card:: 
      :class-card: surface
      :class-body: surface get-started

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-person-walking fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Get Started</h5>
            </div>
         </div>

   .. grid-item-card:: 
      :class-card: surface
      :class-body: surface probing

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-microscope fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Predict</h5>
            </div>
         </div>

   .. grid-item-card:: 
      :class-card: surface
      :class-body: surface steering

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-route fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Control</h5>
            </div>
         </div>

   .. grid-item-card:: 
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-brain fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Understand</h5>
            </div>
         </div>

Welcome! Follow the tutorials below to learn how to conduct interpretability research with :bdg-primary:`nnsight`.

.. grid:: 1 1 2 2
   :class-container: tutorial-card-section
   :gutter: 3

   .. grid-item-card:: 
      :link: notebooks/tutorials/get_started/walkthrough.ipynb
      :class-card: surface
      :class-body: surface get-started

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-person-walking fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Walkthrough</h5>
               <p class="card-text">Learn the basics</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/get_started/start_remote_access.ipynb
      :class-card: surface
      :class-body: surface get-started

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-satellite-dish fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Access LLMs</h5>
               <p class="card-text">Use our hosted models</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/get_started/chat_templates.ipynb
      :class-card: surface
      :class-body: surface get-started

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-list-ol fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Chat Templates</h5>
               <p class="card-text">Format instructions with templates</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/probing/logit_lens.ipynb
      :class-card: surface
      :class-body: surface probing

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-arrow-down-a-z fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Logit Lens</h5>
               <p class="card-text">Decode activations</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/probing/diffusion_lens.ipynb
      :class-card: surface
      :class-body: surface probing

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-camera-retro fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Diffusion Lens</h5>
               <p class="card-text">Explore diffusion model text embedding</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/steering/dict_learning.ipynb
      :class-card: surface
      :class-body: surface steering

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-book-open fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Dictionary Learning</h5>
               <p class="card-text">Sparse autoencoders</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/steering/LoRA_tutorial.ipynb
      :class-card: surface
      :class-body: surface steering

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-sliders fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">LoRA</h5>
               <p class="card-text">Lightweight model adapter</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/causal_mediation_analysis/causal_models_intro.ipynb
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-circle-nodes fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Causal Models</h5>
               <p class="card-text">Intro to causal inference</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/causal_mediation_analysis/causal_mediation_analysis_i.ipynb
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-share-nodes fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Causal Mediation Analysis I</h5>
               <p class="card-text">Use causal mediation to explain AI</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/causal_mediation_analysis/causal_mediation_analysis_ii.ipynb
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-share-nodes fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Causal Mediation Analysis II</h5>
               <p class="card-text">Find causal mediators in LLMs</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/causal_mediation_analysis/activation_patching.ipynb
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-code-pull-request fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Activation Patching</h5>
               <p class="card-text">Causal intervention</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/causal_mediation_analysis/attribution_patching.ipynb
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-diagram-project fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Attribution Patching</h5>
               <p class="card-text">Approximate patching</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/causal_mediation_analysis/DAS.ipynb
      :class-card: surface
      :class-body: surface causal

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-magnifying-glass fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">DAS</h5>
               <p class="card-text">Localizing causal variables</p>
            </div>
         </div>


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   notebooks/tutorials/get_started/walkthrough
   notebooks/tutorials/get_started/start_remote_access
   notebooks/tutorials/get_started/chat_templates

.. toctree::
   :maxdepth: 1
   :caption: Predict

   notebooks/tutorials/probing/logit_lens
   notebooks/tutorials/probing/diffusion_lens

.. toctree::
   :maxdepth: 1
   :caption: Control

   notebooks/tutorials/steering/dict_learning
   notebooks/tutorials/steering/LoRA_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Understand

   Causal Models Intro <notebooks/tutorials/causal_mediation_analysis/causal_models_intro>
   Causal Mediation Analysis I <notebooks/tutorials/causal_mediation_analysis/causal_mediation_analysis_i>
   Causal Mediation Analysis II <notebooks/tutorials/causal_mediation_analysis/causal_mediation_analysis_ii>
   notebooks/tutorials/causal_mediation_analysis/activation_patching
   notebooks/tutorials/causal_mediation_analysis/attribution_patching
   DAS <notebooks/tutorials/causal_mediation_analysis/DAS>
