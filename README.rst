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
Segment
======= ===========


TODOs
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
