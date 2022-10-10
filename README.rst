======
msm_we
======

.. image:: https://github.com/jdrusso/msm_we/actions/workflows/mamba-test.yml/badge.svg
        :target: https://github.com/jdrusso/msm_we/actions/workflows/mamba-test.yml

.. image:: https://github.com/jdrusso/msm_we/actions/workflows/build_docs.yml/badge.svg
        :target: https://github.com/jdrusso/msm_we/actions/workflows/build_docs.yml

.. image:: https://badge.fury.io/py/msm_we.svg
        :target: https://badge.fury.io/py/msm_we

* Authors: John Russo, Jeremy Copperman
* Free software: MIT license
* Documentation: https://msm-we.readthedocs.io .

Background
----------

This is a package for doing history-augmented MSM (haMSM) analysis on weighted ensemble trajectories.

Weighted ensemble data produced from simulations with recycling boundary conditions are naturally in a directional
ensemble.
This means that a history label can be assigned to every trajectory, and an haMSM can be constructed.

Features
--------

* Compute a history-augmented Markov state model from WESTPA weighted ensemble data
* Estimate steady-state distributions
* Estimate flux profiles
* Estimate committors
* WESTPA plugins to automate haMSM construction
* WESTPA plugin to automate bin+allocation optimization


Known Issues
------------

Sometimes, on Python3.7 (and maybe below) the subprocess calls will fail. This may manifest as a silent failure,
followed by hanging (which is very fun to debug!) To fix this, upgrade to Python 3.8+.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
