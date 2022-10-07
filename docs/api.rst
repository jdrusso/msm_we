=============
Documentation
=============


.. autosummary::
   :nosignatures:

   msm_we.msm_we
   msm_we.optimization
   msm_we.westpa_plugins
   msm_we.fpt
   msm_we.ensembles
   msm_we.nmm
   msm_we.utils

haMSM model building and analysis (msm_we.msm_we.modelWE)
---------------------------------------------------------
.. autoclass:: msm_we.msm_we.modelWE

Construction
______________

.. autofunction:: msm_we.msm_we.modelWE.build_analyze_model
.. autofunction:: msm_we.msm_we.modelWE.initialize
.. autofunction:: msm_we.msm_we.modelWE.get_coordSet
.. autofunction:: msm_we.msm_we.modelWE.dimReduce
.. autofunction:: msm_we.msm_we.modelWE.cluster_coordinates
.. autofunction:: msm_we.msm_we.modelWE.get_fluxMatrix
.. autofunction:: msm_we.msm_we.modelWE.organize_fluxMatrix
.. autofunction:: msm_we.msm_we.modelWE.do_block_validation

Analysis
________
.. autofunction:: msm_we.msm_we.modelWE.get_Tmatrix
.. autofunction:: msm_we.msm_we.modelWE.get_steady_state
.. autofunction:: msm_we.msm_we.modelWE.get_steady_state_target_flux
.. autofunction:: msm_we.msm_we.modelWE.get_committor
.. autofunction:: msm_we.msm_we.modelWE.get_flux
.. autofunction:: msm_we.msm_we.modelWE.get_model_clusters
.. autofunction:: msm_we.msm_we.modelWE.get_cluster_centers

Plotting
________
.. autofunction:: msm_we.msm_we.modelWE.plot_flux
.. autofunction:: msm_we.msm_we.modelWE.plot_flux_committor
.. autofunction:: msm_we.msm_we.modelWE.plot_flux_committor_pcoordcolor


Optimization (msm_we.optimization)
----------------------------------

.. automodule:: msm_we.optimization
   :members:


WESTPA Plugins (msm_we.westpa_plugins)
---------------------------------------------------------

.. autoclass:: msm_we.westpa_plugins.augmentation_driver.MDAugmentationDriver
   :members:

.. autoclass:: msm_we.westpa_plugins.hamsm_driver.HAMSMDriver
   :members:

.. autoclass:: msm_we.westpa_plugins.restart_driver.RestartDriver
   :members:

.. autoclass:: msm_we.westpa_plugins.optimization_driver.OptimizationDriver
   :members:

FPT Calculations
----------------

msm_we.fpt
__________
.. automodule:: msm_we.fpt
   :members:

msm_we.ensembles
________________
.. automodule:: msm_we.ensembles
   :members:

msm_we.nmm
__________
.. automodule:: msm_we.nmm
   :members:

msm_we.utils
____________
.. automodule:: msm_we.utils
   :members:
