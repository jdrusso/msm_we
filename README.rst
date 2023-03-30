======
msm_we
======

.. image:: https://github.com/jdrusso/msm_we/actions/workflows/mamba-test.yml/badge.svg?branch=main&event=push
        :target: https://github.com/jdrusso/msm_we/actions/workflows/mamba-test.yml

.. image:: https://readthedocs.org/projects/msm-we/badge/?version=latest
    :target: https://msm-we.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://badge.fury.io/py/msm_we.svg
        :target: https://badge.fury.io/py/msm_we
        
.. image:: https://codecov.io/github/jdrusso/msm_we/branch/main/graph/badge.svg?token=BGVJ3BY6S2 
        :target: https://codecov.io/github/jdrusso/msm_we
        
.. image:: https://zenodo.org/badge/344004587.svg
        :target: https://zenodo.org/badge/latestdoi/344004587

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

- Due to H5py version dependencies, this is currently not compatible with Python 3.10.

- Sometimes, on Python3.7 (and maybe below) the subprocess calls will fail. This may manifest as a silent failure,
  followed by hanging (which is very fun to debug!) To fix this, upgrade to Python 3.8+.

- If running with `$OMP_NUM_THREADS > 1`, Ray parallelism may occasionally silently hang during clustering / fluxmatrix calculations


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
