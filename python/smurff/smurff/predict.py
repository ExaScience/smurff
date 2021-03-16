#!/usr/bin/env python

import numpy as np
from scipy import sparse
import pandas as pd
import os
import os.path
import smurff.matrix_io as mio
from glob import glob
import re
import csv

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser

from .result import Prediction

def read_config_file(file_name, dir_name = None):
    cp = ConfigParser(strict=False)

    if dir_name:
        full_name = os.path.join(dir_name, file_name)
    else:
        full_name = file_name

    with open(full_name) as f:
        try:
            cp.read_file(f, full_name)
        except AttributeError:
            cp.readfp(f, full_name)

    return cp

class Sample:
    @classmethod
    def fromStepFile(cls, file_name, dir_name):
        cp = read_config_file(file_name, dir_name)
        nmodes = int(cp["global"]["num_modes"])
        iter = int(cp["global"]["number"])
        sample = cls(nmodes, iter)

        # predictions, rmse
        sample.pred_stats = dict(read_config_file(cp["predictions"]["pred_state"], dir_name)["global"].items())
        sample.pred_avg = mio.read_matrix(os.path.join(dir_name, cp["predictions"]["pred_avg"]))
        sample.pred_var = mio.read_matrix(os.path.join(dir_name, cp["predictions"]["pred_var"]))

        # latent matrices
        for i in range(sample.nmodes):
            file_name = os.path.join(dir_name, cp["latents"]["latents_" + str(i)])
            U = mio.read_matrix(file_name)
            postMu = None
            postLambda = None

            file_name = cp["latents"]["post_mu_" + str(i)]
            if (file_name != 'none'):
                postMu = mio.read_matrix(os.path.join(dir_name, file_name))

            file_name = cp["latents"]["post_lambda_" + str(i)]
            if (file_name != 'none'):
                postLambda = mio.read_matrix(os.path.join(dir_name, file_name))

            sample.add_latent(U, postMu, postLambda)

            # link matrices (beta) and hyper mus
            beta = np.ndarray((0, 0))
            mu = np.ndarray((0, 0))

            file_name = cp["link_matrices"]["link_matrix_" + str(i)]
            if (file_name != 'none'):
                beta = mio.read_matrix(os.path.join(dir_name, file_name))
            file_name = cp["link_matrices"]["mu_" + str(i)]
            if (file_name != 'none'):
                mu = mio.read_matrix(os.path.join(dir_name, file_name))
                mu = np.squeeze(mu)

            sample.add_beta(beta, mu)

        return sample

    def __init__(self, nmodes, it):
        assert nmodes == 2
        self.nmodes = nmodes
        self.iter = it
        self.latents = []

        self.post_mu = []
        self.post_Lambda = []

        self.betas = []
        self.mus = []

    def check(self):
        for l, b in zip(self.latents, self.betas):
            assert l.shape[0] == self.num_latent()
            assert b.shape[0] == 0 or b.shape[0] == self.num_latent()

    def add_beta(self, b, mu):
        self.betas.append(b)
        self.mus.append(mu)
        self.check()

    def add_latent(self, U, postMu, postLambda):
        self.latents.append(U)
        self.post_mu.append(postMu)
        self.post_Lambda.append(postLambda)
        self.check()

    def num_latent(self):
       return self.latents[0].shape[0]

    def data_shape(self):
       return [u.shape[1] for u in self.latents]

    def beta_shape(self):
       return [b.shape[1] for b in self.betas]

    def predict(self, coords_or_sideinfo=None):
        # for one prediction: einsum(U[:,coords[0]], [0], U[:,coords[1]], [0], ...)
        # for all predictions: einsum(U[0], [0, 0], U[1], [0, 1], U[2], [0, 2], ...)

        cs = coords_or_sideinfo if coords_or_sideinfo is not None else [
            None] * self.nmodes

        operands = []
        for U, mu, c, m in zip(self.latents, self.mus, cs, range(self.nmodes)):
            # predict all in this dimension
            if c is None:
                operands += [U, [0, m+1]]
            else:
                # if side_info was specified for this dimension, we predict for this side_info
                try:  # try to compute sideinfo * beta using dot
                    # compute latent vector from side_info
                    uhat = c.dot(self.betas[m].transpose())
                    if len(uhat.shape) == 2:
                        operands += [uhat + mu, [m+1, 0]]
                    else:
                        operands += [uhat + mu, [0]]
                except AttributeError:  # assume it is a coord
                    # if coords was specified for this dimension, we predict for this coord
                    operands += [U[:, c], [0]]

        return np.einsum(*operands)


class PredictSession:
    """Session for making predictions using a model generated using a :class:`TrainSession`.

    A :class:`PredictSession` can be made directly from a :class:`TrainSession`

    >>> predict_session  = train_session.makePredictSession()

    or from a root file

    >>> predict_session = PredictSession("root.ini")

    """
    @classmethod
    def fromRootFile(cls, root_file):
        return PredictSession(root_file)

    def __init__(self, root_file):
        """Creates a :class:`PredictSession` from a give root file
 
        Parameters
        ----------
        root_file : string
           Name of the root file.
 
        """
        self.root_config = cp = read_config_file(root_file)
        self.root_dir = os.path.dirname(root_file)
        self.options = read_config_file(cp["options"]["options"], self.root_dir)

        self.nmodes = self.options.getint("global", "num_priors")
        assert self.nmodes == 2

        # load only one sample
        for step_name, step_file in cp["steps"].items():
            if (step_name.startswith("sample_step")):
                one_sample = Sample.fromStepFile(step_file, self.root_dir)
                self.num_latent = one_sample.num_latent()
                self.data_shape = one_sample.data_shape()
                self.beta_shape = one_sample.beta_shape()
                return

        raise ValueError("No samples found in " + root_file)

    def samples(self):
        for step_name, step_file in self.root_config["steps"].items():
            if (step_name.startswith("sample_step")):
                yield Sample.fromStepFile(step_file, self.root_dir)

    def lastSample(self):
        steps = list(self.root_config["steps"].items())
        last_step_name, last_step_file = steps[-1]
        if last_step_name.startswith("sample_step"):
            return Sample.fromStepFile(last_step_file, self.root_dir)
        else:
            return None

    def postMuLambda(self, axis):
        sample = self.lastSample()
        if sample is not None:
            postMu = sample.post_mu[axis]
            postLambda = sample.post_Lambda[axis]
            if postMu is not None and postLambda is not None:
                nl = self.num_latent
                assert postLambda.shape[0] == nl * nl
                postLambda = np.reshape(postLambda, (nl, nl, postLambda.shape[1]))
                return postMu, postLambda

        return None, None

    def predictionsYTest(self):
        sample = self.lastSample()
        if sample is not None:
            return sample.pred_avg, sample.pred_var 

        return None, None

    def statsYTest(self):
        sample = self.lastSample()
        if sample is not None:
            return sample.pred_stats 

        return None

    def predict(self, coords_or_sideinfo=None):
        """
        Generate predictions on `coords_or_sideinfo`. Parameters
        specify coordinates of sideinfo/features for each dimension.
        Parameters
        ----------
        operands : tuple 
            A combination of coordindates in the matrix/tensor and/or features you want to use
            to make predictions. `len(coords)` should be equal to number of dimensions in the sample.
            Each element `coords` can be a:
              * int : a single element in this dimension is selected. For example, a
                single row or column in a matrix.
              * :class:`slice` : a slice is selected in this dimension. For example, a number of
                rows or columns in a matrix.
              * None : all elements in this dimension are selected. For example, all
                rows or columns in a matrix.
              * :class:`numpy.ndarray` : 2D numpy array used as dense sideinfo. Each row
                vector is used as side-info.
              * :class:`scipy.sparse.spmatrix` : sparse matrix used as sideinfo. Each row
                vector is used as side-info.

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` of shape `[ N x T1 x T2 x ... ]` where
            N is the number of samples in this `PredictSession` and `T1 x T2 x ...` 
            has the same numer of dimensions as the train data.
        """   
        return np.stack([sample.predict(coords_or_sideinfo) for sample in self.samples()])

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

        for s in self.samples():
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
        for s in self.samples():
            p.add_sample(s.predict(p.coords))

        return p

    def __str__(self):
        dat = (
                -1,
                self.data_shape,
                self.beta_shape,
                self.num_latent
              )
        return "PredictSession with %d samples\n  Data shape = %s\n  Beta shape = %s\n  Num latent = %d" % dat
