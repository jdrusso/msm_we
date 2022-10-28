=============
Documentation
=============


.. autosummary::
   :nosignatures:

   msm_we.modelWE
   msm_we.optimization
   msm_we.fpt
   msm_we.ensembles
   msm_we.nmm
   msm_we.utils
   msm_we.westpa_plugins.augmentation_driver
   msm_we.westpa_plugins.hamsm_driver
   msm_we.westpa_plugins.restart_driver
   msm_we.westpa_plugins.optimization_driver

haMSM model building and analysis (msm_we.modelWE)
---------------------------------------------------------
.. autoclass:: msm_we.modelWE

Construction
______________

.. automethod:: msm_we.modelWE.build_analyze_model
.. automethod:: msm_we.modelWE.initialize
.. automethod:: msm_we.modelWE.get_coordSet
.. automethod:: msm_we.modelWE.dimReduce
.. automethod:: msm_we.modelWE.cluster_coordinates
.. automethod:: msm_we.modelWE.get_fluxMatrix
.. automethod:: msm_we.modelWE.organize_fluxMatrix
.. automethod:: msm_we.modelWE.do_block_validation

Analysis
________
.. automethod:: msm_we.modelWE.get_Tmatrix
.. automethod:: msm_we.modelWE.get_steady_state
.. automethod:: msm_we.modelWE.get_steady_state_target_flux
.. automethod:: msm_we.modelWE.get_committor
.. automethod:: msm_we.modelWE.get_flux
.. automethod:: msm_we.modelWE.get_cluster_centers
.. automethod:: msm_we.modelWE.update_cluster_structures

Plotting
________
.. automethod:: msm_we.modelWE.plot_flux
.. automethod:: msm_we.modelWE.plot_flux_committor
.. automethod:: msm_we.modelWE.plot_flux_committor_pcoordcolor


Optimization (msm_we.optimization)
----------------------------------

.. automodule:: msm_we.optimization
   :members:


WESTPA Plugins (msm_we.westpa_plugins)
---------------------------------------------------------

.. automodule:: msm_we.westpa_plugins.augmentation_driver
   :members: MDAugmentationDriver

.. automodule:: msm_we.westpa_plugins.hamsm_driver
   :members: HAMSMDriver


.. automodule:: msm_we.westpa_plugins.restart_driver
   :members: RestartDriver

.. automodule:: msm_we.westpa_plugins.optimization_driver
   :members: OptimizationDriver

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
