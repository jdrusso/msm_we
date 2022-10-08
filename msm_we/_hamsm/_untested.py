"""Catch-all for code from the original msm_we implementation that is no longer supported, but kept around for legacy
purposes/in case it becomes useful in the future."""
from msm_we.msm_we import modelWE
import pyemma


class ExtendedModelWE(modelWE):

    # For optimized binning
    nB = None
    nW = None
    min_walkers = None

    binMethod = None
    allocationMethod = None

    # vamp_lag = 10
    # vamp_dim = 10
    # nB = 48  # number of bins for optimized WE a la Aristoff
    # nW = 40  # number of walkers for optimized WE a la Aristoff
    # min_walkers = 1  # minimum number of walkers per bin
    # binMethod = "adaptive"  # adaptive for dynamic k-means bin edges, uniform for equal spacing on kh
    # allocationMethod = (
    #     "adaptive"  # adaptive for dynamic allocation, uniform for equal allocation
    # )

    def load_clusters(self, clusterFile):
        """
        Load clusters from a file.

        Updates:
            - `self.clusters`
            - `self.n_clusters`

        Parameters
        ----------
        clusterFile: str
            Filename to load clusters from.

        Returns
        -------
        None
        """

        log.debug(f"Found saved clusters -- loading from {clusterFile}")

        self.clusters = pyemma.load(clusterFile)
        self.n_clusters = np.shape(self.clusters.clustercenters)[0]

    def get_pcoord1D_fluxMatrix(self, n_lag, first_iter, last_iter, binbounds):
        self.n_lag = n_lag
        nBins = binbounds.size - 1
        fluxMatrix = np.zeros((nBins, nBins))
        nI = 0
        f = h5py.File(self.modelName + ".h5", "w")
        dsetName = (
            "/s"
            + str(first_iter)
            + "_e"
            + str(last_iter)
            + "_lag"
            + str(n_lag)
            + "_b"
            + str(nBins)
            + "/pcoord1D_fluxMatrix"
        )
        e = dsetName in f

        # Again, disable this file
        if True or not e:
            dsetP = f.create_dataset(dsetName, np.shape(fluxMatrix))
            for iS in range(first_iter + 1, last_iter + 1):
                if n_lag > 0:
                    fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix(iS, binbounds)
                elif n_lag == 0:
                    log.debug(f"Calling with {iS}, {binbounds}")
                    fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix_lag0(iS, binbounds)
                fluxMatrixI = fluxMatrixI / np.sum(
                    self.weightList
                )  # correct for multiple trees
                fluxMatrix = fluxMatrix + fluxMatrixI
                nI = nI + 1
                dsetP[:] = fluxMatrix / nI
                dsetP.attrs["iter"] = iS
            fluxMatrix = fluxMatrix / nI
        elif e:
            dsetP = f[dsetName]
            fluxMatrix = dsetP[:]
            try:
                nIter = dsetP.attrs["iter"]
            except Exception as e:
                nIter = first_iter + 1
                fluxMatrix = np.zeros((nBins, nBins))
                log.error(e)
            nI = 1
            if nIter < last_iter:
                for iS in range(nIter, last_iter + 1):
                    if n_lag > 0:
                        # TODO: Is this even implemented..?
                        fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix(iS, binbounds)
                    if n_lag == 0:
                        fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix_lag0(
                            iS, binbounds
                        )
                    fluxMatrix = fluxMatrix + fluxMatrixI
                    nI = nI + 1
                    dsetP[:] = fluxMatrix / nI
                    dsetP.attrs["iter"] = iS
                fluxMatrix = fluxMatrix / nI
        f.close()
        self.pcoord1D_fluxMatrix = fluxMatrix

    def get_iter_pcoord1D_fluxMatrix_lag0(self, n_iter, binbounds):
        """
        Compute a flux-matrix in the space of the 1D progress coordinate, at a given iteration.

        Parameters
        ----------
        n_iter: integer
            Iteration to compute flux matrix for

        binbounds: array-like
            Array of progress coordinate bin boundaries

        Returns
        -------
        The fluxmatrix at iteration n_iter.
        """

        # log.info(n_iter)
        # log.info(binbounds)

        log.debug("iteration " + str(n_iter) + ": solving fluxmatrix \n")

        self.load_iter_data(n_iter)

        weightList = self.weightList.copy()

        num_segments_in_iter = np.shape(self.weightList)[0]

        nBins = binbounds.size - 1
        fluxMatrix = np.zeros((nBins, nBins))

        # Lists of all parent and child pcoords
        pcoord0 = self.pcoord0List[:, 0]
        pcoord1 = self.pcoord1List[:, 0]

        # Assign parent/child pcoords to bins
        bins0 = np.digitize(pcoord0, binbounds)
        bins1 = np.digitize(pcoord1, binbounds)

        for seg_idx in range(num_segments_in_iter):
            # I THINK WHAT'S HAPPENING HERE IS:
            # The lowest binbound provided here should be smaller than the smallest possible
            #   value in the trajectory.
            # I.e., because of the way bins are defined for WE, you might have bin bounds like
            #   [0,1,2,3], where x>3 is in the basis and 0<x<1 is in the target. However, digitize
            #   would assign a point of 0.5, which should be in the target, to index 1.
            # I think this -1 corrects for that.

            from_bin_index = bins0[seg_idx] - 1
            to_bin_index = bins1[seg_idx] - 1

            # Set the weight of any segment that jumps more than 12 bins to 0?
            #   This seems super risky, and also originally this function didn't copy weightList, so it
            #   modified weightList in state for anything that runs after.
            # In particular, I think that's bad because if you have more than 12 bins, going from the target to
            #   the basis is going to be set to 0 by this logic.
            # So, I'm going to disable this for now...
            if False and np.abs(from_bin_index - to_bin_index) > 12:
                weightList[seg_idx] = 0.0

            fluxMatrix[from_bin_index, to_bin_index] = (
                fluxMatrix[from_bin_index, to_bin_index] + weightList[seg_idx]
            )

        return fluxMatrix

    def get_model_clusters(self):
        """
        Used by get_iter_aristoffian(). Untested and un-debugged, use at your own risk.

        Updates:
        - self.model_clusters
        """
        # define new clusters from organized flux matrix corresponding to model

        log.critical(
            "This function is untested, and may rely on other untested parts of this code. Use with extreme caution."
        )

        clustercenters = np.zeros((self.n_clusters + 2, self.ndim))
        clustercenters[0 : self.n_clusters, :] = self.clusters.clustercenters
        if self.dimReduceMethod == "none":
            coords = np.array([self.basis_coords, self.reference_coord])
            clustercenters[self.n_clusters :, :] = self.reduceCoordinates(
                coords
            )  # add in basis and target
            self.model_clusters = clustering.AssignCenters(
                self.reduceCoordinates(clustercenters[self.originalClusters, :]),
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        if self.dimReduceMethod == "pca" or self.dimReduceMethod == "vamp":
            coords = np.array(
                [
                    self.reduceCoordinates(self.basis_coords),
                    self.reduceCoordinates(self.reference_coord),
                ]
            )
            clustercenters[self.n_clusters :, :] = np.squeeze(
                coords
            )  # add in basis and target

            self.model_clusters = clustering.AssignCenters(
                clustercenters[self.originalClusters, :],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )

    def get_hflux(self, conv):
        convh = conv
        convf = conv
        max_iters = 50000
        nTargets = self.indTargets.size
        indNotTargets = np.setdiff1d(range(self.nBins), self.indTargets)
        Mt = self.Tmatrix.copy()
        dconvh = 1.0e100
        dconvf = 1.0e100
        fTotal = np.zeros((self.nBins, 1))
        fSSp = 0.0
        hp = np.zeros_like(fTotal)
        N = 1
        while dconvh > convh or dconvf > convf and N < max_iters:
            f = np.zeros((self.nBins, 1))
            for i in range(self.nBins):
                Jt = 0.0
                Pt = Mt[i, :]
                for j in range(nTargets):
                    jj = self.indTargets[j]
                    Jt = Jt + np.sum(
                        np.multiply(
                            Pt[0, indNotTargets],
                            Mt[indNotTargets, jj * np.ones_like(indNotTargets)],
                        )
                    )
                f[i, 0] = Jt / self.tau
            fTotal = fTotal + f
            fSS = np.mean(f[indNotTargets, 0])
            ht = fTotal - N * fSS
            dconvh = np.max(np.abs(hp - ht)) / np.max(ht)
            dconvf = np.abs(fSS - fSSp) / fSS
            sys.stdout.write(
                "N="
                + str(N)
                + " dh: "
                + str(dconvh)
                + " df: "
                + str(dconvf)
                + " Jss:"
                + str(fSS)
                + "\n"
            )
            hp = ht.copy()
            fSSp = fSS
            self.h = ht.copy()
            Mt = np.matmul(Mt, self.Tmatrix)
            N = N + 1

    def get_model_aristoffian(self):
        kh = np.matmul(self.Tmatrix, self.h)
        kh_sq = np.power(kh, 2)
        hsq = np.power(self.h, 2)
        k_hsq = np.matmul(self.Tmatrix, hsq)
        varh = k_hsq - kh_sq
        # val=np.sqrt(varh)
        self.varh = varh
        self.kh = kh

    def get_model_steady_state_aristoffian(self):
        nB = int(self.nB)
        if self.binMethod == "adaptive":
            self.kh_clusters = coor.cluster_kmeans(self.kh, k=nB, metric="euclidean")
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
            khbins_centers_unique, ind_unique = np.unique(
                khbins_centers, return_index=True
            )
            if khbins_centers_unique.size != nB:
                khbins = np.squeeze(
                    np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
                )  # equal spacing if not enough for k-means
                khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
        elif self.binMethod == "uniform":
            khbins = np.squeeze(
                np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
            )  # equal spacing if not enough for k-means
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "log_uniform":
            transformedBins = np.geomspace(
                np.abs(np.min(self.kh)) / np.max(self.kh),
                1.0 + 2.0 * np.abs(np.min(self.kh)) / np.max(self.kh),
                self.nB + 1,
            )
            khbins_binEdges_log = transformedBins * np.max(self.kh) - 2.0 * np.abs(
                np.min(self.kh)
            )
            khbins = khbins_binEdges_log  # equal log-spacing
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "optimized":
            try:
                khbins_centers = np.loadtxt("khbins_binCenters.dat")
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            except Exception as e:
                log.error(e)
                sys.stdout.write(
                    "khbins (khbins_binCenters.dat) not found: initializing\n"
                )
                self.get_initial_khbins_equalAlloc()
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            if not hasattr(self, "kh_clusters"):
                sys.stdout.write("giving up: log uniform kh bins")
                self.get_initial_khbins()
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
        dtraj_kh_clusters = self.kh_clusters.assign(self.kh)
        alloc = np.zeros(
            nB
        )  # get bin objective function, value and allocation over set of bins
        value = np.zeros(nB)
        bin_kh_var = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            if indBin[0].size == 0:
                alloc[i] = 0.0
                bin_kh_var[i] = 0.0
            else:
                # n = indBin[0].size
                bin_kh_var[i] = np.var(self.kh[indBin])
                wt = np.sum(self.pSS[indBin])
                vw = np.sum(np.multiply(self.pSS[indBin] / wt, self.varh[indBin]))
                alloc[i] = wt * (vw) ** 0.5
                value[i] = vw**0.5
        if self.allocationMethod == "uniform":
            alloc = np.ones_like(alloc)
        alloc = alloc / np.sum(alloc)
        self.alloc = alloc
        # base_walkers=self.min_walkers*np.ones_like(alloc)
        # nBase=np.sum(base_walkers)
        # nAdapt=self.nW-nBase
        # if nAdapt<0:
        #    nAdapt=0
        # walkers=np.round(alloc*nAdapt)
        # walkers=walkers+base_walkers
        # indZero=np.where(walkers==0.0)
        # walkers[indZero]=1.0
        # walkers=walkers.astype(int)
        # binEdges=np.zeros(self.nB+1)
        # binEdges[0]=-np.inf
        # binEdges[-1]=np.inf
        # ind=np.argsort(self.kh_clusters.clustercenters[:,0]) #note sorting makes kh_clusters indexes differen
        # self.khbins_binCenters=self.kh_clusters.clustercenters[ind,0]
        # binEdges[1:-1]=0.5*(self.khbins_binCenters[1:]+self.khbins_binCenters[0:-1])
        # self.khbins_binEdges=binEdges
        # self.walkers_per_bin=walkers[ind]
        # self.bin_kh_var=bin_kh_var[ind]
        gamma = self.alloc.copy()  # asymptotic particle distribution in bins
        # asymptotic particle distribution after mutation
        rho = np.zeros_like(gamma)
        rhov = np.zeros((self.nB, self.nB))
        for v in range(self.nB):
            indBinv = np.where(dtraj_kh_clusters == v)
            wv = np.sum(self.pSS[indBinv])
            sys.stdout.write("sum v: " + str(v) + "\n")
            for u in range(self.nB):
                indBinu = np.where(dtraj_kh_clusters == u)
                for p in indBinv[0]:
                    for q in indBinu[0]:
                        rhov[u, v] = (
                            rhov[u, v]
                            + self.alloc[v] * (self.pSS[p] / wv) * self.Tmatrix[p, q]
                        )
        rho = np.sum(rhov, 1)
        pOccupied = 1.0 - np.power(1.0 - rho, self.nW)
        nOccupied = nB - np.sum(np.power(1.0 - rho, self.nW))
        nAdditional = (self.nW - nOccupied) * self.alloc
        nT = nAdditional + pOccupied
        # nT=np.zeros(nB)
        # for i in range(nB):
        #    nT[i]=np.max(np.array([1,nAdditional[i]+pOccupied[i]]))
        bin_mutV = np.zeros(nB)
        bin_selV = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            wi = np.sum(self.pSS[indBin])
            bin_mutV[i] = ((wi**2) / (nT[i])) * np.sum(
                np.multiply(self.pSS[indBin] / wi, self.varh[indBin])
            )
            bin_selV[i] = ((wi**2) / (nT[i])) * np.sum(
                np.multiply(self.pSS[indBin] / wi, np.power(self.kh[indBin], 2))
                - np.power(np.multiply(self.pSS[indBin] / wi, self.kh[indBin]), 2)
            )
        self.binObjective = np.sum(bin_mutV + bin_selV)
        binEdges = np.zeros(self.nB + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        ind = np.argsort(
            self.kh_clusters.clustercenters[:, 0]
        )  # note sorting makes kh_clusters indexes different
        self.khbins_binCenters = self.kh_clusters.clustercenters[ind, 0]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        # self.walkers_per_bin=walkers[ind]
        self.bin_kh_var = bin_kh_var[ind]
        base_walkers = self.min_walkers * np.ones_like(alloc)
        nBase = nOccupied  # estimated from occupied bins a la Aristoff notes, was np.sum(base_walkers)
        nAdapt = self.nW - nBase
        if nAdapt < 0:
            nAdapt = 0
        walkers = np.round(alloc * nAdapt)
        walkers = walkers + base_walkers
        indZero = np.where(walkers == 0.0)
        walkers[indZero] = 1.0
        walkers = walkers.astype(int)
        self.walkers_per_bin = walkers[ind]
        self.bin_mutV = bin_mutV[ind]
        self.bin_selV = bin_selV[ind]
        self.nOccupancySS = nT[ind]
        self.nOccupied = nOccupied
        self.nAdapt = nAdapt
        self.rhomutation = rho[ind]
        self.value = value

    def get_initial_khbins(self):  # log-uniform kh bins
        transformedBins = np.geomspace(
            np.abs(np.min(self.kh)) / np.max(self.kh),
            1.0 + 2.0 * np.abs(np.min(self.kh)) / np.max(self.kh),
            self.nB + 1,
        )
        khbins_binEdges_log = transformedBins * np.max(self.kh) - 2.0 * np.abs(
            np.min(self.kh)
        )
        khbins = khbins_binEdges_log  # equal log-spacing
        khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
        self.kh_clusters = clustering.AssignCenters(
            khbins_centers[:, np.newaxis],
            metric="euclidean",
            stride=1,
            n_jobs=None,
            skip=0,
        )
        binEdges = np.zeros(self.nB + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        ind = np.argsort(
            self.kh_clusters.clustercenters[:, 0]
        )  # note sorting makes kh_clusters indexes different
        self.khbins_binCenters = self.kh_clusters.clustercenters[ind, 0]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        np.savetxt("khbins_binCenters.dat", self.khbins_binCenters)

    def get_initial_khbins_equalAlloc(self):  # kh bins approximately of equal value
        if not hasattr(self, "kh"):
            self.get_model_aristoffian()
        binMethod_use = self.binMethod
        allocationMethod_use = self.allocationMethod
        nB_use = self.nB
        self.binMethod = "uniform"
        self.allocationMethod = "adaptive"
        points = np.linspace(0, 1, self.nB)
        # resN=np.round(abs(np.max(self.kh)/np.min(np.abs(self.kh)))).astype(int)
        resN = 10000
        self.nB = resN
        self.get_model_steady_state_aristoffian()
        dist = self.alloc.copy()
        dist = dist / np.sum(dist)
        dist = np.cumsum(dist)
        dist_unique, ind_unique = np.unique(dist, return_index=True)
        kh_unique = self.khbins_binCenters[ind_unique]
        xB = np.zeros_like(points)
        for i in range(xB.size):
            indm = np.argmin(np.abs(dist_unique - points[i]))
            xB[i] = kh_unique[indm]
            dist_unique[indm] = np.inf
        khbins_centers = xB.copy()
        self.nB = nB_use
        self.binMethod = binMethod_use
        self.allocationMethod = allocationMethod_use
        self.kh_clusters = clustering.AssignCenters(
            khbins_centers[:, np.newaxis],
            metric="euclidean",
            stride=1,
            n_jobs=None,
            skip=0,
        )
        binEdges = np.zeros(self.nB + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        ind = np.argsort(
            self.kh_clusters.clustercenters[:, 0]
        )  # note sorting makes kh_clusters indexes different
        self.khbins_binCenters = self.kh_clusters.clustercenters[ind, 0]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        np.savetxt("khbins_binCenters.dat", self.khbins_binCenters)

    def get_bin_kh_var(self, x):
        nB = self.nB
        self.kh_clusters = clustering.AssignCenters(
            x[:, np.newaxis], metric="euclidean", stride=1, n_jobs=None, skip=0
        )
        dtraj_kh_clusters = self.kh_clusters.assign(self.kh)
        # alloc=np.zeros(nB) #get bin objective function, value and allocation over set of bins
        bin_kh_var = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            if indBin[0].size == 0:
                # alloc[i]=0.0
                bin_kh_var[i] = 0.0
            else:
                # n = indBin[0].size
                bin_kh_var[i] = np.var(self.kh[indBin])
                # wt=np.sum(self.pSS[indBin])
                # vw=np.sum(np.multiply(self.pSS[indBin]/wt,self.varh[indBin]))
                # alloc[i]=wt*(vw)**.5
        self.bin_kh_var = bin_kh_var
        self.total_bin_kh_var = np.sum(bin_kh_var)
        return self.total_bin_kh_var

    def get_bin_total_var(self, x):
        # nB = self.nB
        self.kh_clusters = clustering.AssignCenters(
            x[:, np.newaxis], metric="euclidean", stride=1, n_jobs=None, skip=0
        )
        self.binMethod = "optimized"
        self.get_model_steady_state_aristoffian()
        return self.binObjective

    def get_iter_aristoffian(self, iter):

        log.critical(
            "This function is untested, and may rely on other untested parts of this code. Use with extreme caution."
        )

        self.load_iter_data(iter)
        if not hasattr(self, "model_clusters"):
            self.get_model_clusters()
        #        if self.pcoord_is_kh:
        # wt=np.sum(self.pSS[indBin])
        # vw=np.sum(np.multiply(self.pSS[indBin]/wt,self.varh[indBin]))
        # alloc[i]=wt*(vw)**.5
        #            self.khList=np.array(self.pcoord1List[:,1]) #kh is pcoord 2 from optimized WE sims
        #            self.khList=self.khList[:,np.newaxis]
        #        else:
        self.load_iter_coordinates()
        dtraj_iter = self.model_clusters.assign(
            self.reduceCoordinates(self.cur_iter_coords)
        )
        kh_iter = self.kh[dtraj_iter]
        self.khList = np.array(kh_iter[:, 0])  # get k-means bins defined over walkers
        nB = self.nB
        khList_unique = np.unique(self.khList)
        if khList_unique.size > 2.0 * nB and self.binMethod == "adaptive":
            self.kh_clusters = coor.cluster_kmeans(
                khList_unique, k=nB, metric="euclidean"
            )
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
            khbins_centers_unique, ind_unique = np.unique(
                khbins_centers, return_index=True
            )
            if khbins_centers_unique.size != nB:
                khbins = np.squeeze(
                    np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
                )  # equal spacing if not enough for k-means
                khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
        elif self.binMethod == "uniform":
            khbins = np.squeeze(
                np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
            )  # equal spacing if not enough for k-means
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "log_uniform":
            transformedBins = np.geomspace(
                np.abs(np.min(self.kh)) / np.max(self.kh),
                1.0 + 2.0 * np.abs(np.min(self.kh)) / np.max(self.kh),
                self.nB,
            )
            khbins_binEdges_log = transformedBins * np.max(self.kh) - 2.0 * np.abs(
                np.min(self.kh)
            )
            khbins = khbins_binEdges_log  # equal log-spacing
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "optimized":
            try:
                khbins_centers = np.loadtxt("khbins_binCenters.dat")
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            except Exception as e:
                log.error(e)
                log.debug("khbins (khbins_binCenters.dat) not found: initializing\n")
                self.get_initial_khbins_equalAlloc()
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            if not hasattr(self, "kh_clusters"):
                sys.stdout.write("giving up: log uniform kh bins")
                self.get_initial_khbins()
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
        dtraj_kh_clusters = self.kh_clusters.assign(self.khList)
        varh_iter = self.varh[dtraj_iter]
        alloc = np.zeros(
            nB
        )  # get bin objective function, value and allocation over set of bins
        bin_kh_var = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            if indBin[0].size == 0:
                alloc[i] = 0.0
                bin_kh_var[i] = 0.0
            else:
                # n = indBin[0].size
                bin_kh_var[i] = np.var(self.khList[indBin])
                wt = np.sum(self.weightList[indBin])
                vw = np.sum(np.multiply(self.weightList[indBin], varh_iter[indBin]))
                alloc[i] = (wt * vw) ** 0.5
        if self.allocationMethod == "uniform":
            alloc = np.ones_like(alloc)
        alloc = alloc / np.sum(alloc)
        self.alloc = alloc
        base_walkers = self.min_walkers * np.ones_like(alloc)
        nBase = np.sum(base_walkers)
        if hasattr(self, "nAdapt"):
            nAdapt = self.nAdapt
        else:
            nAdapt = self.nW - nBase
        if nAdapt < 0:
            nAdapt = 0
        walkers = np.round(alloc * nAdapt)
        walkers = walkers + base_walkers
        indZero = np.where(walkers == 0.0)
        walkers[indZero] = 1.0
        walkers = walkers.astype(int)
        khbins_centers = self.kh_clusters.clustercenters[:, 0]
        khbins_centers_unique, ind_unique = np.unique(khbins_centers, return_index=True)
        walkers = walkers[ind_unique]
        bin_kh_var = bin_kh_var[ind_unique]
        binEdges = np.zeros(khbins_centers_unique.size + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        # ind=np.argsort(khbins_centers_unique) #note sorting makes kh_clusters indexes different
        self.khbins_binCenters = khbins_centers_unique  # [ind]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        self.walkers_per_bin = walkers  # [ind]
        self.bin_kh_var = bin_kh_var  # [ind]
        self.binObjective = np.sum(bin_kh_var)

    def write_iter_kh_pcoord(self):  # grab coordinates from WE traj_segs folder
        nS = self.nSeg
        if not hasattr(self, "model_clusters"):
            self.get_model_clusters()
        self.load_iter_coordinates()  # post coordinates
        dtraj_iter = self.model_clusters.assign(
            self.reduceCoordinates(self.cur_iter_coords)
        )
        kh_iter = self.kh[dtraj_iter]
        khList1 = np.array(kh_iter[:, 0])  # post pcoord
        self.load_iter_coordinates0()  # pre coordinates
        dtraj_iter = self.model_clusters.assign(
            self.reduceCoordinates(self.cur_iter_coords)
        )
        kh_iter = self.kh[dtraj_iter]
        khList0 = np.array(kh_iter[:, 0])  # pre pcoord
        westFile = self.fileList[self.westList[0]]
        dataIn = h5py.File(westFile, "a")
        pcoords = np.zeros(
            (0, 2, 2)
        )  # this is explicitly set up for p1 (target,basis def) and p2 (kh-aristoffian)
        for iS in range(self.nSeg):
            westFile = self.fileList[self.westList[iS]]
            pcoord = np.zeros((1, 2, 2))
            pcoord[0, 0, 0] = self.pcoord0List[iS, 0]
            pcoord[0, 1, 0] = self.pcoord1List[iS, 0]
            pcoord[0, 0, 1] = khList0[iS, 0]
            pcoord[0, 1, 1] = khList1[iS, 0]
            pcoords = np.append(pcoords, pcoord, axis=0)
            try:
                if iS > 0:
                    if self.westList[iS] != self.westList[iS - 1] and iS < nS - 1:
                        dsetName = "/iterations/iter_%08d/pcoord" % int(self.n_iter)
                        del dataIn[dsetName]
                        dset = dataIn.create_dataset(
                            dsetName, np.shape(pcoords[:-1, :, :])
                        )
                        dset[:] = pcoords[:-1, :, :]
                        dataIn.close()
                        pcoords = np.zeros(
                            (0, 2, 2)
                        )  # this is explicitly set up for p1 (target,basis def) and p2 (kh-aristoffian)
                        pcoords = np.append(pcoords, pcoord, axis=0)
                        dataIn = h5py.File(westFile, "a")
                    elif iS == nS - 1:
                        dsetName = "/iterations/iter_%08d/pcoord" % int(self.n_iter)
                        del dataIn[dsetName]
                        dset = dataIn.create_dataset(dsetName, np.shape(pcoords))
                        dset[:] = pcoords
                        sys.stdout.write(
                            "pcoords for iteration "
                            + str(self.n_iter)
                            + " overwritten\n"
                        )
                        dataIn.close()
            except Exception as e:
                log.error(e)
                log.error(
                    "error overwriting pcoord from "
                    + westFile
                    + " , iter "
                    + str(self.n_iter)
                    + " segment "
                    + str(self.segindList[iS])
                    + "\n"
                )

    def get_warps_from_parent(self, first_iter, last_iter):
        """
        Get all warps and weights over a range of iterations.

        Parameters
        ----------
        first_iter: int
            First iteration in range.
        last_iter: int
            Last iteration in range.

        Returns
        -------
        warpedWeights: list
            List of weights for each warp.

        """
        warpedWeights = []
        for iS in range(first_iter + 1, last_iter + 1):
            self.load_iter_data(iS + 1)
            self.get_seg_histories(2)
            parentList = self.seg_histories[:, 1]
            warpList = np.where(parentList < 0)
            warpedWeights.append(self.weightList[warpList])
        return warpedWeights

    def get_warps_from_pcoord(
        self, first_iter, last_iter
    ):  # get all warps and weights over set of iterations
        warpedWeights = []
        for iS in range(first_iter, last_iter + 1):
            self.load_iter_data(iS)
            pcoord = self.pcoord1List[:, 0]
            # warpList = np.where(pcoord < self.WEtargetp1)
            warpList = np.where(self.is_WE_target(pcoord))
            warpedWeights.append(self.weightList[warpList])
            meanJ = (
                np.mean(self.weightList[warpList]) / self.tau / np.sum(self.weightList)
            )
            sys.stdout.write("Jdirect: " + str(meanJ) + " iter: " + str(iS) + "\n")
        return warpedWeights

    def get_direct_target_flux(self, first_iter, last_iter, window):
        nIterations = last_iter - first_iter
        Jdirect = np.zeros(nIterations - 1)
        f = h5py.File(self.modelName + ".h5", "a")
        dsetName = (
            "/s"
            + str(first_iter)
            + "_e"
            + str(last_iter)
            + "_w"
            + str(window)
            + "/Jdirect"
        )
        e = dsetName in f
        if not e:
            warpedWeights = self.get_warps_from_pcoord(first_iter, last_iter)
            self.warpedWeights = warpedWeights
            JdirectTimes = np.zeros(nIterations - 1)
            for iS in range(nIterations - 1):
                end = iS + 1
                start = iS - window
                if start < 0:
                    start = 0
                nI = end - start
                warpedWeightsI = np.array([])
                for i in range(start, end):
                    warpedWeightsI = np.append(warpedWeightsI, warpedWeights[i])
                nWarped = warpedWeightsI.size
                particles = (nWarped * warpedWeightsI) / nI
                Jdirect[iS] = np.mean(particles)
                JdirectTimes[iS] = (first_iter + iS) * self.tau
            Jdirect = Jdirect / self.tau
            dsetP = f.create_dataset(dsetName, np.shape(Jdirect))
            dsetP[:] = Jdirect
            dsetName = (
                "/s"
                + str(first_iter)
                + "_e"
                + str(last_iter)
                + "_w"
                + str(window)
                + "/JdirectTimes"
            )
            dsetP = f.create_dataset(dsetName, np.shape(JdirectTimes))
            dsetP[:] = JdirectTimes
        elif e:
            dsetP = f[dsetName]
            Jdirect = dsetP[:]
            dsetName = (
                "/s"
                + str(first_iter)
                + "_e"
                + str(last_iter)
                + "_w"
                + str(window)
                + "/JdirectTimes"
            )
            dsetP = f[dsetName]
            JdirectTimes = dsetP[:]
        f.close()
        self.Jdirect = Jdirect / self.n_data_files  # correct for number of trees
        self.JdirectTimes = JdirectTimes

        def evolve_probability(
            self, nEvolve, nStore
        ):  # iterate nEvolve times, storing every nStore iterations, initial condition at basis
            nIterations = np.ceil(nEvolve / nStore).astype(int) + 1
            self.nEvolve = nEvolve
            self.nStore = nStore
            Mss = self.Tmatrix
            nBins = self.nBins
            binCenters = self.binCenters
            probBasis = np.zeros((1, self.nBins))
            probBasis[0, self.indBasis] = 1.0
            pSS = probBasis.copy()
            pSSPrev = np.ones_like(pSS)
            iT = 1
            probTransient = np.zeros((nIterations, nBins))  # while dConv>.00001:
            probTransient[0, :] = probBasis
            for i in range(nEvolve):
                pSS = np.matmul(pSS, Mss)
                dConv = np.sum(np.abs(pSS - pSSPrev))
                pSSPrev = pSS.copy()
                if i % nStore == 0:
                    sys.stdout.write("SS conv: " + str(dConv) + " iter: " + str(i))
                    try:
                        plt.plot(
                            binCenters,
                            np.squeeze(pSS),
                            "-",
                            color=plt.cm.Greys(float(i) / float(nEvolve)),
                        )
                        plt.yscale("log")
                    except Exception as e:
                        log.error(e)
                        try:
                            plt.plot(
                                binCenters,
                                pSS.A1,
                                "-",
                                color=plt.cm.Greys(float(i) / float(nEvolve)),
                            )
                        # ????? why these nested excepts? What's so fragile here? Maybe the shape of pSS?
                        except Exception as e2:
                            log.error(e2)
                            pass
                    probTransient[iT, :] = np.squeeze(pSS)
                    iT = iT + 1
            probTransient = probTransient[0:iT, :]
            self.probTransient = probTransient
            pSS = np.squeeze(np.array(pSS))
            self.pSS = pSS / np.sum(pSS)
            try:
                plt.pause(4)
                plt.close()
            except Exception as e:
                log.error(e)
                pass

        def evolve_probability2(
            self, nEvolve, nStore
        ):  # iterate nEvolve times, storing every nStore iterations, initial condition spread for everything at RMSD higher than basis
            nIterations = np.ceil(nEvolve / nStore).astype(int) + 1
            self.nEvolve = nEvolve
            self.nStore = nStore
            Mss = self.Tmatrix
            nBins = self.nBins
            binCenters = self.binCenters
            probBasis = np.zeros((1, self.nBins))
            probBasis[
                0, self.indBasis[0] :
            ] = 1.0  # assign initial probability to everything at RMSD higher than the basis, for case when nothing observed leaving exact basis
            probBasis = probBasis / np.sum(probBasis)
            pSS = probBasis.copy()
            pSSPrev = np.ones_like(pSS)
            iT = 1
            probTransient = np.zeros((nIterations, nBins))  # while dConv>.00001:
            probTransient[0, :] = probBasis
            for i in range(nEvolve):
                pSS = np.matmul(pSS, Mss)
                dConv = np.sum(np.abs(pSS - pSSPrev))
                pSSPrev = pSS.copy()
                if i % nStore == 0:
                    sys.stdout.write("SS conv: " + str(dConv) + " iter: " + str(i))
                    try:
                        plt.plot(
                            binCenters,
                            np.squeeze(pSS),
                            "-",
                            color=plt.cm.Greys(float(i) / float(nEvolve)),
                        )
                        plt.yscale("log")
                    except Exception as e:
                        log.error(e)
                        try:
                            plt.plot(
                                binCenters,
                                pSS.A1,
                                "-",
                                color=plt.cm.Greys(float(i) / float(nEvolve)),
                            )
                        except Exception as e2:
                            log.error(e2)
                            pass
                    # plt.ylim([1e-100,1])
                    # plt.title(str(iT)+' of '+str(nIterations))
                    # plt.pause(.1)
                    probTransient[iT, :] = np.squeeze(pSS)
                    iT = iT + 1
            probTransient = probTransient[0:iT, :]
            self.probTransient = probTransient
            pSS = np.squeeze(np.array(pSS))
            self.pSS = pSS / np.sum(pSS)
            try:
                plt.pause(4)
                plt.close()
            except Exception as e:
                log.error(e)
                pass

        def evolve_probability_from_initial(
            self, p0, nEvolve, nStore
        ):  # iterate nEvolve times, storing every nStore iterations, initial condition provided
            nIterations = np.ceil(nEvolve / nStore).astype(int) + 1
            self.nEvolve = nEvolve
            self.nStore = nStore
            Mss = self.Tmatrix
            nBins = self.nBins
            binCenters = self.binCenters
            probBasis = np.zeros((1, self.nBins))
            if np.shape(probBasis)[1] == np.shape(p0)[0]:
                probBasis[0, :] = p0
            else:
                probBasis = p0
            pSS = probBasis.copy()
            pSSPrev = np.ones_like(pSS)
            iT = 1
            probTransient = np.zeros((nIterations, nBins))  # while dConv>.00001:
            probTransient[0, :] = probBasis
            for i in range(nEvolve):
                pSS = np.matmul(pSS, Mss)
                dConv = np.sum(np.abs(pSS - pSSPrev))
                pSSPrev = pSS.copy()
                if i % nStore == 0:
                    sys.stdout.write("SS conv: " + str(dConv) + " iter: " + str(i))
                    try:
                        plt.plot(
                            binCenters,
                            np.squeeze(pSS),
                            "-",
                            color=plt.cm.Greys(float(i) / float(nEvolve)),
                        )
                        plt.yscale("log")
                    except Exception as e:
                        log.error(e)
                        try:
                            plt.plot(
                                binCenters,
                                pSS.A1,
                                "-",
                                color=plt.cm.Greys(float(i) / float(nEvolve)),
                            )
                        except Exception as e2:
                            log.error(e2)
                            pass
                    probTransient[iT, :] = np.squeeze(pSS)
                    iT = iT + 1
            probTransient = probTransient[0:iT, :]
            self.probTransient = probTransient
            pSS = np.squeeze(np.array(pSS))
            self.pSS = pSS / np.sum(pSS)
            try:
                plt.pause(4)
                plt.close()
            except Exception as e:
                log.error(e)
                pass
