import logging

import numpy as np
import scipy.sparse as sp

from .helper import SparseTensor, FixedNoise, SampledNoise
from .wrapper import NoiseConfig, StatusItem, PythonSession
from .predict import PredictSession


class TrainSession(PythonSession):
    """Class for doing a training run in smurff

    A simple use case could be:

    >>> trainSession = smurff.TrainSession(burnin = 5, nsamples = 5)
    >>> trainSession.setTrain(Ydense)
    >>> trainSession.run()

    Attributes
    ----------

    priors: list, where element is one of { "normal", "normalone", "macau", "macauone", "spikeandslab" }
        The type of prior to use for each dimension

    num_latent: int
        Number of latent dimensions in the model

    burnin: int
        Number of burnin samples to discard
    
    nsamples: int
        Number of samples to keep

    num_threads: int
        Number of OpenMP threads to use for model building

    verbose: {0, 1, 2}
        Verbosity level for C++ library

    seed: float
        Random seed to use for sampling

    save_name: path
        HDF5 filename to store the samples.

    save_freq: int
        - N>0: save every Nth sample
        - N==0: never save a sample
        - N==-1: save only the last sample
        
    checkpoint_freq: int
        Save the state of the trainSession every N seconds.

    """
    #
    # construction functions
    #
    def __init__(self,
        priors           = ["normal", "normal"],
        num_latent       = None,
        num_threads      = None,
        burnin           = None,
        nsamples         = None,
        seed             = None,
        threshold        = None,
        verbose          = None,
        save_name        = None,
        save_freq        = None,
        checkpoint_freq  = None,
        ):

        super().__init__()

        if priors is not None:          self.setPriorTypes(priors)
        if num_latent is not None:      self.setNumLatent(num_latent)
        if num_threads is not None:     self.setNumThreads(num_threads)
        if burnin is not None:          self.setBurnin(burnin)
        if nsamples is not None:        self.setNSamples(nsamples)
        if seed is not None:            self.setRandomSeed(seed)
        if threshold is not None:       self.setThreshold(threshold)
        if verbose is not None:         self.setVerbose(verbose)
        if save_name is not None:       self.setSaveName(save_name)
        if save_freq is not None:       self.setSaveFreq(save_freq)
        if checkpoint_freq is not None: self.setCheckpointFreq(checkpoint_freq)


    def addTrainAndTest(self, Y, Ytest = None, noise = FixedNoise(), is_scarce = True):
        self.setTrain(Y, noise, is_scarce)

        if Ytest is not None:
            self.setTest(Ytest)

    def setTrain(self, Y, noise = FixedNoise(), is_scarce = True):
        """Adds a train and optionally a test matrix as input data to this TrainSession

        Parameters
        ----------

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Train matrix/tensor 

        noise : :class: `NoiseConfig`
            Noise model to use for `Y`

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        """
        
        super().setTrain(Y, noise, is_scarce)
       
    def addSideInfo(self, mode, Y, noise = SampledNoise(), direct = True):
        """Adds fully known side info, for use in with the macau or macauone prior

        mode : int
            dimension to add side info (rows = 0, cols = 1)

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix
            Side info matrix/tensor 
            Y should have as many rows in Y as you have elemnts in the dimension selected using `mode`.
            Columns in Y are features for each element.

        noise : :class: `NoiseConfig`
            Noise model to use for `Y`
        
        direct : boolean
            - When True, uses a direct inversion method. 
            - When False, uses a CG solver 

            The direct method is only feasible for a small (< 100K) number of features.

        """
        super().addSideInfo(mode, Y, noise, direct)

    def addPropagatedPosterior(self, mode, mu, Lambda):
        """Adds mu and Lambda from propagated posterior

        mode : int
            dimension to add side info (rows = 0, cols = 1)

        mu : :class: `numpy.ndarray` matrix
            mean matrix  
            mu should have as many rows as `num_latent`
            mu should have as many columns as size of dimension `mode` in `train`

        Lambda : :class: `numpy.ndarray` matrix
            co-variance matrix  
            Lambda should be shaped like K x K x N 
            Where K == `num_latent` and N == dimension `mode` in `train`
        """
        if len(Lambda.shape) == 3:
            assert Lambda.shape[0] == self.num_latent
            assert Lambda.shape[1] == self.num_latent
            Lambda = Lambda.reshape(self.num_latent * self.num_latent, Lambda.shape[2], order='F')

        self.addPropagatedPosterior(mode, mu, Lambda)


    def addData(self, pos, Y, noise = FixedNoise(), is_scarce = False):
        """Stacks more matrices/tensors next to the main train matrix.

        pos : shape
            Block position of the data with respect to train. The train matrix/tensor
            has implicit block position (0, 0). 

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Data matrix/tensor to add

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        noise : :class: `NoiseConfig`
            Noise model to use for `Y`
        
        """
        if isinstance(Y, np.ndarray):
            # dense array or matrix
            super().addData(pos, Y, noise)
        elif sp.issparse(Y):
            # sparse/scarce scipy.sparse matrix
            super().addData(pos, Y.tocsr(), noise, is_scarce)
        elif isinstance(Y, SparseTensor):
            # sparse/scarce scipy.sparse tensor
            super().addData(pos, Y, noise, is_scarce)
        else:
            raise TypeError("Unsupported type for addData: {}. We support numpy.ndarray, scipy.sparce matrix or SparseTensora.".format(Y))


    # 
    # running functions
    #

    def init(self):
        """Initializes the `TrainSession` after all data has been added.

        You need to call this method befor calling :meth:`step`, unless you call :meth:`run`

        Returns
        -------
        :class:`StatusItem` of the trainSession.

        """

        super().init()
        logging.info(self)
        return self.getStatus()

    def step(self):
        """Does on sampling or burnin iteration.

        Returns
        -------
        - When a step was executed: :class:`StatusItem` of the trainSession.
        - After the last iteration, when no step was executed: `None`.

        """
        not_done = super().step()
        
        if self.interrupted():
            raise KeyboardInterrupt

        if not_done:
            return self.getStatus()
        else:
            return None

    def run(self):
        """Equivalent to:

        .. code-block:: python
        
            self.init()
            while self.step():
                pass
        """
        self.init()
        while self.step():
            pass

        return self.getTestPredictions()

    def makePredictSession(self):
        """Makes a :class:`PredictSession` based on the model
           that as built in this `TrainSession`.

        """
        return PredictSession(self.getSaveName())