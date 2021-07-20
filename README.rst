======
msm_we
======

.. image:: https://github.com/jdrusso/msm_we/actions/workflows/main.yml/badge.svg
        :target: https://github.com/jdrusso/msm_we/actions/workflows/main.yml


Code for Markov state modeling of weighted ensemble trajectories.


* Free software: MIT license
* Documentation: https://jdrusso.github.io/msm_we/ .


Features
--------

* TODO

Jargon
------

======= ===========
Word     Definition
======= ===========
Warp     When a particle is recycled from a sink state to a source state
------- -----------
Segment  A trajectory from a single WE iteration
======= ===========


Code development TODOs
-----
*  Fold :code:`collectCoordinates.py` functionality into :code:`msm_we` proper.
*  Does :code:`reduceCoordinates(self, coords)` actually need to be a mandatory user-override?
*  Provide option for user to override :code:`dimReduce()` in case they want to do something other than PCA/VAMP
*  Add TICA option to :code:`dimReduce()`
*  Add monkey-patching directly into :code:`msm_we` code. I.e., add a function like :code:`register_featurization(func)`
    which takes the featurization function as input and overloads :code:`msm_we.processCoordinates()` with it.
    Right now, this has to be done manually.

Code understanding TODOs
-----
*  When does :code:`segindList` not contain a consecutive list of integers from 1 to n_segs?
*  What triggers the exception being handled in :code:`get_transition_data()`?
*  What are "warps"? :code:`get_transition_data()`
    * Warps are recycling events
*  What exactly is being stored in :code:`seg_histories`?


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
