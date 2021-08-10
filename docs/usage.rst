=====
Usage
=====

To use msm_we in a project::

    import msm_we


The basic workflow of this module is as follows.

Model building and preparation
------------------------------

1. Run a weighted ensemble simulation, and produce a :code:`west.h5` file.

2. Augment :code:`west.h5` with the full coordinate information in each :code:`<iterXXX/auxdata/coord>`.

    :code:`msm_we/scripts/collectCoordinates/collectCoordinates.py` has an example of this.
    This functionality will likely eventually be rolled into the main package.

3. Define the :code:`processCoordinates(self, coords)` function and monkey-patch it in.

    This function is responsible for featurization. It should take in an array of all coordinates,
    and produce an array of feature-coordinates.

    .. code-block:: python

        # A trivial example processCoordinates
        def processCoordinates(self, coords):
            return coords

        # Monkey-patch, i.e. replace the placeholder processCoordinates
        #   in msm_we.modelWE with your function
        msm_we.modelWE.processCoordinates = processCoordinates


    It's important to do this monkey-patching at the module level, i.e. on :code:`msm_we.modelWE`, rather
    than on an instance of a :code:`msm_we.modelWE` object.

4. Create the model object.

    .. code-block:: python

        model = msm_we.modelWE()

5. Initialize the model.

    .. code-block:: python

        model.initialize(h5_glob, reference_structure_file, model_name)


    :code:`h5_glob` is a list of paths to your WESTPA h5 files.

    :code:`reference_structure_file` is a file containing a topology describing your system.

    :code:`model_name` is what it sounds like, and is used to label various output files.

5. Manually set the following parameters.

    .. code-block:: python

        model.WEtargetp1_bounds = [
            #lower bound of the target state in progress coordinate,
            #upper bound of the target state in progress coordinate,
        ]

        model.WEbasisp1_bounds = [
            #lower bound of the basis state in progress coordinate,
            #upper bound of the basis state in progress coordinate,
        ]

        model.pcoord_ndim0 = # Number of pcoord dimensions
        model.dimReduceMethod = # Dimensionality reduction method. Must be "pca", "vamp", or "none"


6. Load all coords and pcoords up to the last iteration you want to use for analysis with

    .. code-block:: python

        model.get_coordSet(last_iter)

7. Prepare dimensionality reduction transformer by running

    .. code-block:: python

        model.dimReduce()

8. Do clustering with

    .. code-block:: python

        model.cluster_coordinates(n_clusters)

9. Create the flux matrix with

    .. code-block:: python

        model.get_fluxMatrix(lag, first_iter, last_iter)

    a. Clean disconnected states and sort the flux matrix with

    .. code-block:: python

        model.organize_fluxMatrix()

Analysis
--------

10. Normalize the flux matrix to produce a transition matrix with

    .. code-block:: python

        model.get_Tmatrix()

11. Obtain steady-state distribution with

    .. code-block:: python

        model.get_steady_state_algebraic()

12. Update cluster structures

    .. code-block:: python

        model.update_cluster_structures()

13. Obtain steady-state target flux with

    .. code-block:: python

        model.get_steady_state_target_flux()
