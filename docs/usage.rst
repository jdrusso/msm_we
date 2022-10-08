==================
Installation/Usage
==================


To install :code:`msm_we`:

.. code-block:: bash

    cd </path/to/msm_we>

    conda env create -f environment.yml

or, to install it into an existing conda environment:

.. code-block:: bash

    cd </path/to/msm_we>

    conda env update --name <your WESTPA environment> --file environment.yml

.. highlight:: python

To use :code:`msm_we` in a project::

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

    ::

        # A trivial example processCoordinates
        def processCoordinates(self, coords):
            # Do some stuff in here
            # This is NOT a complete example!
            return coords

        # Monkey-patch, i.e. replace the placeholder processCoordinates
        #   in msm_we.modelWE with your function
        msm_we.modelWE.processCoordinates = processCoordinates


    It's important to do this monkey-patching at the module level, i.e. on the :class:`msm_we.modelWE`
    class itself, rather than on an instance of the class.

4. Create the model object.

    .. code-block:: python

        model = msm_we.modelWE()

5. Initialize the model.

    .. code-block:: python

        model.initialize(file_paths, reference_structure_file, model_name,
                        basis_pcoord_bounds, target_pcoord_bounds,
                        dim_reduce_method, tau, pcoord_ndim)


    :code:`file_paths` is a list of paths to your WESTPA h5 files.

    :code:`reference_structure_file` is a file containing a topology describing your system.

    :code:`model_name` is what it sounds like, and is used to label various output files.

    :code:`basis_pcoord1_bounds` is a list of [[pcoord0 lower bound, pcoord1 upper bound],
    [pcoord1 lower bound, pcoord1 upper bound], ...] in pcoord-space for the basis state

    :code:`target_pcoord1_bounds` is a list of [[pcoord0 lower bound, pcoord1 upper bound],
    [pcoord1 lower bound, pcoord1 upper bound], ...] in pcoord-space for the target state

    :code:`dim_reduce_method` is the dimensionality reduction method ("pca", "vamp", or "none")

    :code:`tau` is the resampling time, or the length of one WE iteration in physical units.

    :code:`pcoord_ndim` is the dimensionality of the progress coordinate.

6. Load all coords and pcoords up to the last iteration you want to use for analysis with

    .. code-block:: python

        model.get_iterations()
        model.get_coordSet(last_iter, streaming)

    where :code:`last_iter` is the number of iterations you have (AKA, the last iteration it'll load data from)
    and :code:`streaming` enables streaming data processing, which allows large datasets to fit in memory at the cost of
    a (nominally) small performance hit.

7. Prepare dimensionality reduction transformer by running

    .. code-block:: python

        model.dimReduce()

8. Do clustering with

    .. code-block:: python

        model.cluster_coordinates(n_clusters, streaming,
            first_cluster_iter, use_ray, stratified,
            **_cluster_args)

    :code:`n_clusters` is the total number of clusters if :code:`stratified=False`, or the number of clusters per bin if :code:`stratified=True`.

    :code:`streaming` is whether or not to stream over the data.

    :code:`first_cluster_iter` is the first iteration used for building the cluster centers (which may be desirable to exclude
    initial burn-in iterations).

    :code:`use_ray` enables parallelization with the Ray work manager. If enabled, a Ray cluster must be initialized by
    the user *before calling this*.

    :code:`stratified` enables stratified clustering instead of aggregate clustering. In stratified clustering,
    clustering is done independently within each WE bin. This is strongly recommended to ensure your clusters provide a
    good fine-grained description of your system.

    **Note**: At time of writing, stratified clustering implies :code:`streaming=True, use_ray=True` and will enable this
    with a warning if they are not set.

    Any additional keyword arguments will be provided directly to the clustering function through :code:`**_cluster_args`.

9. Create the flux matrix with

    .. code-block:: python

        model.get_fluxMatrix(lag, first_iter, last_iter, use_ray)

    :code:`lag` is the lag-time used for model-building. Currently, only 0 is supported, which corresponds to looking at
    transitions from each parent segment directly to its child.

    :code:`first_iter`, and :code:`last_iter` are the first and last iteration to use when computing the flux matrix.
    Note that excluding many iterations may result in limited connectivity of the flux matrix, as early events may have
    provided critical transitions between WE bins that may not be otherwise sampled.

    :code:`use_ray` enables parallelization with the Ray work manager. If enabled, a Ray cluster must be initialized by
    the user *before calling this*.

    a. Clean disconnected states and sort the flux matrix with

    .. code-block:: python

        model.organize_fluxMatrix(use_ray)


Analysis
--------

10. Normalize the flux matrix to produce a transition matrix with

    .. code-block:: python

        model.get_Tmatrix()

11. Obtain steady-state distribution with

    .. code-block:: python

        model.get_steady_state()

    Note: This may fail or encounter difficulties for datasets where no target flux has been obtained.
    This can happen with either incomplete sampling to your target state, or with equilibrium data.
    This is because it uses the flux estimate as a convergence criterion.
    If the flux is 0, then it's not meaningful to  look at convergence of 0, so it'll just run
    for the maximum number of iterations. You can specify :code:`max_iters=1` to avoid unnecessary
    iteration, or you can use :meth:`~msm_we.modelWE.get_steady_state_algebraic`.

12. Update cluster structures

    .. code-block:: python

        model.update_cluster_structures()

13. Obtain steady-state target flux with

    .. code-block:: python

        model.get_steady_state_target_flux()

Streaming
---------

:code:`msm_we` supports streaming dimensionality reduction and clustering when dimensionality reduction is
done through PCA or not done.

Streaming dimensionality reduction is automatically done for PCA.

To use streaming clustering, pass :code:`streaming=True` to :code:`cluster_coordinates()`.

Streaming is not supported for VAMP, because I don't know of a streaming implementation of VAMP dimensionality reduction.

Parallelism
-----------

:code:`msm_we` supports parallelism of many "slow" parts of model-building -- namely, clustering, discretization, and
flux matrix calculations. This is done through the Ray work manager.

Before invoking any function with :code:`use_ray=True`, a Ray work manager must be initialized on the machine running
the analysis. In the simplest case, this can just be

.. code-block:: python

    import ray
    ray.init()

:code:`msm_we` will connect to whatever Ray instance is running on the machine the analysis is being performed on.
However, this can be used on a cluster to initialize a Ray cluster with workers on a number of nodes, and the :code:`msm_we`
running on the same node as the Ray head.

Model-building, in one command
------------------------------------

The above steps are all wrapped in a single call by :meth:`~msm_we.modelWE.build_analyze_model`,
which can be called like

.. code-block:: python

    from msm_we import modelWE

    model = modelWE()

    model.build_analyze_model(
        file_paths=we_h5filenames,
        ref_struct=basis_ref_dict,
        modelName=msm_we_modelname,
        basis_pcoord_bounds=pcoord_bounds["basis"],
        target_pcoord_bounds=pcoord_bounds["target"],
        dimreduce_method="pca",
        n_clusters=msm_we_n_clusters
    )

This performs all the steps manually outlined above, up to flux calculation.
Additionally, this can split up your data to do block validation.

Most arguments to :meth:`~msm_we.modelWE.initialize` can also be passed to
:meth:`~msm_we.modelWE.build_analyze_model`.

See the documentation for additional arguments that can be passed.

Optimization
------------

To use :mod:`msm_we.optimization`, first construct an haMSM as outlined above.

The goal of the optimization algorithm is to group haMSM microbins into WE bins, in "some optimal" way.
In this case, we optimize for minimizing flux variance.

.. code-block:: python

    import msm_we.optimization as mo

    discrepancy, variance = mo.solve_discrepancy(
        tmatrix = transition_matrix,
        pi = steady_state_distribution,
        B = target_state_indices
    )

    # This is a list with an element for each MSM microbin, which is the integer index of the
    #   WE bin it's assigned to.
    # In other words, microstate_assignments[microbin_index] == WE bin index of that microbin
    microstate_assignments = mo.get_uniform_mfpt_bins(
        variance, discrepancy, steady_state_distribution, n_active_we_bins
    )

    # Add entries for the basis/target states, since MSM-WE sets those as the last two clusters
    microstate_assignments = np.concatenate(
        [microstate_assignments, [n_active_bins - 2, n_active_bins - 1]]
    )

    # Create the new bin mapper for WESTPA
    we_bin_mapper = mo.OptimizedBinMapper(
        n_active_we_bins,
        # In case the pcoord is extended, this is the original pcoord dimensionality
        n_pcoord_dims,
        # The original, non-Optimized BinMapper that WESTPA was run with.
        #   Used for stratified clustering
        base_mapper,
        microstate_assignments,
        # If the pcoord was extended, pcoord boundaries are in the original pcoord space
        basis_pcoord_bounds,
        target_pcoord_bounds,
        stratified_clusterer
    )

WESTPA Integration
------------------

Coming soon...
