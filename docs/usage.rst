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

3. Define :code:`processCoordinates(self, coords)` and :code:`reduceCoordinates(self, coords)` functions.

    :code:`processCoordinates(self, coords)` is responsible for featurization. It should take in an array of all coordinates,
    and produce an array of feature-coordinates.

    :code:`reduceCoordinates(self, coords)` handles running the dimensionality reduction. :code:`msm_we.dimReduce()`
    defines the actual coordinate transformer object, in :code:`msm_we.coordinates`. However, :code:`reduceCoordinates()`
    actually runs that transformation on a set of input data.

    An example of how this is currently done, to read a couple functions from a Python file with path
    :code:`override_file` and monkey-patch them in:

    .. code-block:: python

        # We monkey-patch at the module level rather than just override the function in the instanced object
        #   so that the functions retain access to self.
        msm_we.modelWE.processCoordinates = user_overrides.processCoordinates
        msm_we.modelWE.reduceCoordinates = user_overrides.reduceCoordinates

4. Create the model object.

.. code-block:: python

    model = msm_we.modelWE()

5. Initialize the model.

.. code-block:: python

    model.initialize(h5_glob, reference_structure_file, basis_structure_file, model_name)

6. Load all coords and pcoords into  :code:`model.all_coords` and :code:`model.pcoordSet` respectively.

7. Prepare dimensionality reduction transformer by running

.. code-block:: python

    model.dimReduce()

8. Do clustering with

.. code-block:: python

    model.cluster_coordinates(n_clusters)

9. Create the flux matrix with

.. code-block:: python

    model.get_fluxMatrix(lag, first_iter, last_iter)

a. (Optionally) Clean disconnected states and sort the flux matrix with :code:`model.organize_fluxMatrix`

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
