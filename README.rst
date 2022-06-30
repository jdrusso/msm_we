======
msm_we
======

.. image:: https://github.com/jdrusso/msm_we/actions/workflows/run_tests.yml/badge.svg
        :target: https://github.com/jdrusso/msm_we/actions/workflows/run_tests.yml

.. image:: https://github.com/jdrusso/msm_we/actions/workflows/build_docs.yml/badge.svg
        :target: https://github.com/jdrusso/msm_we/actions/workflows/build_docs.yml

Code for Markov state modeling of weighted ensemble trajectories.

* Authors: John Russo, Jeremy Copperman
* Free software: MIT license
* Documentation: https://jdrusso.github.io/msm_we/ .


Features
--------

* Compute a history-augmented Markov state model from WESTPA weighted ensemble data
* Estimate steady-state distributions
* Estimate flux profiles
* Estimate committors


Known Issues
------------

Sometimes, on Python3.7 (and maybe below) the subprocess calls will fail. This may manifest as a silent failure,
followed by hanging (which is very fun to debug!) To fix this, upgrade to Python 3.8+.


Code development TODOs
----------------------
*  Provide option for user to override :code:`dimReduce()` in case they want to do something other than PCA/VAMP
*  Add TICA option to :code:`dimReduce()`


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
