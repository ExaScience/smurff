#!/usr/bin/env python

import numpy as np
import h5sparse

from .result import Prediction

class Sample:
    def __init__(self, h5_file, name, num_latent):
        self.h5_group = h5_file[name]
        self.no = int(self.h5_group.attrs["number"])
        self.nmodes = h5_file["config/options"].attrs['num_priors']
        self.num_latent = num_latent

    def predStats(self):
        # rmse , auc
        return self.h5_group["predictions"].attrs

    def predAvg(self):
        return self.h5_group["predictions/pred_avg"][()]

    def predVar(self):
        return self.h5_group["predictions/pred_var"][()]

    def lookup_mode(self, templ, mode):
        return self.h5_group[templ % mode][()]

    def lookup_modes(self, templ):
        return [ self.lookup_mode(templ, i) for i in range(self.nmodes) ]

    def latents(self):
        return self.lookup_modes("latents/latents_%d")

    def betas(self):
        return self.lookup_modes("link_matrices/link_matrix_%d")
        
    def mus(self):
        return self.lookup_modes("link_matrices/mu_%d")

    def beta_shape(self):
        return [b.shape[1] for b in self.betas()]

    def postMuLambda(self, mode):
        mu = self.lookup_mode("latents/post_mu_%d", mode)
        Lambda = self.lookup_mode("latents/post_lambda_%d", mode)
        assert Lambda.shape[1] == self.num_latent * self.num_latent
        Lambda = Lambda.reshape(Lambda.shape[0], self.num_latent, self.num_latent)

        return mu, Lambda

    def predict(self, coords_or_sideinfo=None):
        # for one prediction: einsum(U[:,coords[0]], [0], U[:,coords[1]], [0], ...)
        # for all predictions: einsum(U[0], [0, 0], U[1], [0, 1], U[2], [0, 2], ...)

        cs = coords_or_sideinfo if coords_or_sideinfo is not None else [
            None] * self.nmodes

        operands = []
        for U, mu, c, m in zip(self.latents(), self.mus(), cs, range(self.nmodes)):
            # predict all in this dimension
            if c is None:
                operands += [U, [m+1, 0]]
            else:
                # if side_info was specified for this dimension, we predict for this side_info
                try:  # try to compute sideinfo * beta using dot
                    # compute latent vector from side_info
                    uhat = c.dot(self.betas()[m]).transpose()
                    mu = np.squeeze(mu)
                    uhat = np.squeeze(uhat) 
                    operands += [uhat + mu, [0]]
                except AttributeError:  # assume it is a coord
                    # if coords was specified for this dimension, we predict for this coord
                    operands += [U[c, :], [0]]

        return np.einsum(*operands)


class PredictSession:
    """TrainSession for making predictions using a model generated using a :class:`TrainSession`.

    A :class:`PredictSession` can be made directly from a :class:`TrainSession`

    >>> predict_session  = train_session.makePredictSession()

    or from a root file

    >>> predict_session = PredictSession("root.ini")

    """
    def __init__(self, h5_fname):
        """Creates a :class:`PredictSession` from a given HDF5 file
 
        Parameters
        ----------
        h5_fname : string
           Name of the HDF5 file.
 
        """
        self.h5_file = h5sparse.File(h5_fname, 'r')
        self.options = self.h5_file["config/options"].attrs
        self.nmodes = int(self.options['num_priors'])
        self.num_latent = int(self.options["num_latent"])
        self.data_shape = self.h5_file["config/train/data"].shape

        self.samples = []
        for name in self.h5_file.keys():
            if not name.startswith("sample_"):
                continue

            sample = Sample(self.h5_file, name, self.num_latent)
            self.samples.append(sample)
            self.beta_shape = sample.beta_shape()

        if len(self.samples) == 0:
            raise ValueError("No samples found in " + h5_fname)

        self.samples.sort(key=lambda x: x.no)
        self.num_samples = len(self.samples)

    def lastSample(self):
        return self.samples[-1]

    def postMuLambda(self, mode):
        return self.lastSample().postMuLambda(mode)

    def predictionsYTest(self):
        return self.lastSample().predAvg(), self.lastSample().predVar()

    def statsYTest(self):
        return self.lastSample().predStats()

    def predict(self, coords_or_sideinfo=None):
        return np.stack([sample.predict(coords_or_sideinfo) for sample in self.samples])

    def predict_all(self):
        """Computes the full prediction matrix/tensor.

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` of shape `[ N x T1 x T2 x ... ]` where
            N is the number of samples in this `PredictSession` and `T1 x T2 x ...` 
            is the shape of the train data.

        """        
        return self.predict()

    def predict_some(self, test_matrix):
        """Computes prediction for all elements in a sparse test matrix

        Parameters
        ----------
        test_matrix : scipy sparse matrix
            Coordinates and true values to make predictions for

        Returns
        -------
        list 
            list of :class:`Prediction` objects.

        """        
        predictions = Prediction.fromTestMatrix(test_matrix)

        for s in self.samples:
            for p in predictions:
                p.add_sample(s.predict(p.coords))

        return predictions

    def predict_one(self, coords_or_sideinfo, value=float("nan")):
        """Computes prediction for one point in the matrix/tensor

        Parameters
        ----------
        coords_or_sideinfo : tuple of coordinates and/or feature vectors
        value : float, optional
            The *true* value for this point

        Returns
        -------
        :class:`Prediction`
            The prediction

        """
        p = Prediction(coords_or_sideinfo, value)
        for s in self.samples:
            p.add_sample(s.predict(p.coords))

        return p

    def __str__(self):
        dat = (len(self.samples), self.data_shape, self.beta_shape, self.num_latent)
        return "PredictSession with %d samples\n  Data shape = %s\n  Beta shape = %s\n  Num latent = %d" % dat
