##
#
# Cython file that bridges between the smurff python code and the C++ code
# 
# This file will be compiled via CMake
#


from libc.stdint cimport *
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.string cimport string

from cython.operator cimport dereference as deref, preincrement as inc

from Config cimport *
from NoiseConfig cimport *
from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig
from SideInfoConfig cimport *
from SessionFactory cimport SessionFactory

from ISession cimport ISession
from ResultItem cimport ResultItem
from StatusItem cimport StatusItem

cimport numpy as np
import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers
import tempfile
import os
import logging

from .helper import SparseTensor, PyNoiseConfig, StatusItem as PyStatusItem
from .prepare import make_train_test, make_train_test_df
from .result import Prediction
from .predict import PredictSession

DENSE_MATRIX_TYPES  = (np.ndarray, np.matrix, )
SPARSE_MATRIX_TYPES = (sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix, )

MATRIX_TYPES = DENSE_MATRIX_TYPES + SPARSE_MATRIX_TYPES

DENSE_TENSOR_TYPES  = (np.ndarray, )
SPARSE_TENSOR_TYPES = (SparseTensor, )

TENSOR_TYPES = DENSE_TENSOR_TYPES + SPARSE_TENSOR_TYPES

ALL_TYPES = MATRIX_TYPES + TENSOR_TYPES

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
    idx = np.where(np.isnan(Y.data) == False)[0]
    return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

cdef MatrixConfig* prepare_sparse_matrix(X, NoiseConfig noise_config, is_scarce) except +:
    if not isinstance(X, SPARSE_MATRIX_TYPES):
        raise ValueError("Matrix must be either coo, csr or csc (from scipy.sparse)")

    X = X.tocoo(copy = False)
    X = remove_nan(X)

    cdef np.ndarray[uint32_t] irows = X.row.astype(np.uint32, copy=False)
    cdef np.ndarray[uint32_t] icols = X.col.astype(np.uint32, copy=False)
    cdef np.ndarray[double] vals = X.data.astype(np.double, copy=False)

    # Get begin and end pointers from numpy arrays
    cdef uint32_t* irows_begin = &irows[0]
    cdef uint32_t* irows_end = irows_begin + irows.shape[0]
    cdef uint32_t* icols_begin = &icols[0]
    cdef uint32_t* icols_end = icols_begin + icols.shape[0]
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]

    # Create vectors from pointers
    cdef vector[uint32_t]* rows_vector_ptr = new vector[uint32_t]()
    rows_vector_ptr.assign(irows_begin, irows_end)
    cdef vector[uint32_t]* cols_vector_ptr = new vector[uint32_t]()
    cols_vector_ptr.assign(icols_begin, icols_end)
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)

    cdef shared_ptr[vector[uint32_t]] rows_vector_shared_ptr = shared_ptr[vector[uint32_t]](rows_vector_ptr)
    cdef shared_ptr[vector[uint32_t]] cols_vector_shared_ptr = shared_ptr[vector[uint32_t]](cols_vector_ptr)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)

    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), rows_vector_shared_ptr, cols_vector_shared_ptr, vals_vector_shared_ptr, noise_config, is_scarce)
    return matrix_config_ptr

cdef MatrixConfig* prepare_dense_matrix(X, NoiseConfig noise_config) except +:
    cdef np.ndarray[np.double_t] vals = np.squeeze(np.asarray(X.flatten(order='F')))
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)
    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), vals_vector_shared_ptr, noise_config)
    return matrix_config_ptr

cdef shared_ptr[SideInfoConfig] prepare_sideinfo(side_info, NoiseConfig noise_config, tol, direct) except +:
    if isinstance(side_info, SPARSE_MATRIX_TYPES):
        side_info_config_matrix = prepare_sparse_matrix(side_info, noise_config, False)
    elif isinstance(side_info, DENSE_MATRIX_TYPES) and len(side_info.shape) == 2:
        side_info_config_matrix = prepare_dense_matrix(side_info, noise_config)
    else:
        error_msg = "Unsupported side info matrix type: {}".format(side_info)
        raise ValueError(error_msg)

    cdef shared_ptr[SideInfoConfig] side_info_config_ptr = make_shared[SideInfoConfig]()
    side_info_config_ptr.get().setSideInfo(shared_ptr[MatrixConfig](side_info_config_matrix))
    side_info_config_ptr.get().setTol(tol)
    side_info_config_ptr.get().setDirect(direct)
    return side_info_config_ptr

cdef TensorConfig* prepare_dense_tensor(tensor, NoiseConfig noise_config) except +:
    if not isinstance(tensor, DENSE_TENSOR_TYPES):
        error_msg = "Unsupported dense tensor data type: {}".format(tensor)
        raise ValueError(error_msg)

    cdef np.ndarray[np.double_t] vals = np.squeeze(np.asarray(tensor.flatten(order='F')))
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)

    cdef vector[uint64_t] cpp_dims_vector
    cpp_dims_vector = tensor.shape

    cdef TensorConfig* tensor_config_ptr = new TensorConfig(
         make_shared[vector[uint64_t]](cpp_dims_vector), vals_vector_shared_ptr, noise_config)
    return tensor_config_ptr        

cdef TensorConfig* prepare_sparse_tensor(tensor, NoiseConfig noise_config, is_scarse) except +:
    shape = tensor.shape
    df = tensor.data

    cdef vector[uint64_t] cpp_dims_vector
    cdef vector[uint32_t] cpp_columns_vector
    cdef vector[double] cpp_values_vector

    idx_column_names = list(filter(lambda c: df[c].dtype==np.int64 or df[c].dtype==np.int32, df.columns))
    val_column_names = list(filter(lambda c: df[c].dtype==np.float32 or df[c].dtype==np.float64, df.columns))

    if len(val_column_names) != 1:
        error_msg = "tensor has {} float columns but must have exactly 1 value column.".format(len(val_column_names))
        raise ValueError(error_msg)

    idx = [i for c in idx_column_names for i in np.array(df[c], dtype=np.int32)]
    val = np.array(df[val_column_names[0]],dtype=np.float64)

    cpp_dims_vector = shape
    cpp_columns_vector = idx
    cpp_values_vector = val

    return new TensorConfig(
            make_shared[vector[uint64_t]](cpp_dims_vector),
            make_shared[vector[uint32_t]](cpp_columns_vector),
            make_shared[vector[double]](cpp_values_vector),
            noise_config,
            is_scarse)

cdef shared_ptr[TensorConfig] prepare_auxdata(data, pos, is_scarce, NoiseConfig noise_config) except +:
    cdef TensorConfig* aux_data_config
    cdef vector[int] cpos = pos

    if isinstance(data, DENSE_MATRIX_TYPES) and len(data.shape) == 2:
        aux_data_config = prepare_dense_matrix(data, noise_config)
    elif isinstance(data, SPARSE_MATRIX_TYPES):
        aux_data_config = prepare_sparse_matrix(data, noise_config, is_scarce)
    elif isinstance(data, DENSE_TENSOR_TYPES):
        aux_data_config = prepare_dense_tensor(data, noise_config)
    elif isinstance(data, SPARSE_TENSOR_TYPES):
        aux_data_config = prepare_sparse_tensor(data, noise_config, is_scarce)
    else:
        error_msg = "Unsupported tensor type: {}".format(type(data))
        raise ValueError(error_msg)

    aux_data_config.setPos(cpos)

    return shared_ptr[TensorConfig](aux_data_config)

cdef (shared_ptr[TensorConfig], shared_ptr[TensorConfig]) prepare_train_and_test(train, test, NoiseConfig noise_config, bool is_scarce) except +:

    # Check train data type
    if (not isinstance(train, ALL_TYPES)):
        error_msg = "Unsupported train data type: {}".format(type(train))
        raise ValueError(error_msg)

    # Check test data type
    if test is not None:
        if (not isinstance(test, ALL_TYPES)):
            error_msg = "Unsupported test data type: {}".format(type(test))
            raise ValueError(error_msg)

    # Check train and test data for mismatch
    if test is not None:
        if isinstance(train, MATRIX_TYPES) and isinstance(test, MATRIX_TYPES):
            pass
        elif isinstance(train, TENSOR_TYPES) and isinstance(test, TENSOR_TYPES):
            pass
        else:
            error_msg = "Train and test data must be the same type: {} != {}".format(type(train), type(test))
            raise ValueError(error_msg)
        if isinstance(train, MATRIX_TYPES) and train.shape != test.shape:
            raise ValueError("Train and test data must be the same shape: {} != {}".format(train.shape, test.shape))
        if isinstance(train, SPARSE_TENSOR_TYPES) and train.ndim != test.ndim:
            raise ValueError("Train and test data must have the same number of dimensions: {} != {}".format(train.ndim, test.ndim))

    cdef TensorConfig* train_config = NULL
    cdef TensorConfig* test_config = NULL

    # Prepare train data
    if isinstance(train, DENSE_MATRIX_TYPES) and len(train.shape) == 2:
        train_config = prepare_dense_matrix(train, noise_config)
    elif isinstance(train, SPARSE_MATRIX_TYPES):
        train_config = prepare_sparse_matrix(train, noise_config, is_scarce)
    elif isinstance(train, DENSE_TENSOR_TYPES) and len(train.shape) > 2:
        train_config = prepare_dense_tensor(train, noise_config)
    elif isinstance(train, SPARSE_TENSOR_TYPES):
        train_config = prepare_sparse_tensor(train, noise_config, is_scarce)
    else:
        error_msg = "Unsupported train data type or shape: {}".format(type(train))
        raise ValueError(error_msg)

    # Prepare test data
    if test is not None:
        if isinstance(test, DENSE_MATRIX_TYPES) and len(test.shape) == 2:
            test_config = prepare_dense_matrix(test, noise_config)
        elif isinstance(test, SPARSE_MATRIX_TYPES):
            test_config = prepare_sparse_matrix(test, noise_config, True)
        elif isinstance(test, DENSE_TENSOR_TYPES) and len(test.shape) > 2:
            test_config = prepare_dense_tensor(test, noise_config)
        elif isinstance(test, SPARSE_TENSOR_TYPES):
            test_config = prepare_sparse_tensor(test, noise_config, True)
        else:
            error_msg = "Unsupported test data type: {} or shape {}".format(type(test), test.shape)
            raise ValueError(error_msg)

    return shared_ptr[TensorConfig](train_config), shared_ptr[TensorConfig](test_config)

cdef NoiseConfig prepare_noise_config(py_noise_config):
    """converts a PyNoiseConfig object to a C++ NoiseConfig object"""
    cdef NoiseConfig n
    n.setNoiseType(py_noise_config.noise_type.encode('UTF-8'))
    n.setPrecision(py_noise_config.precision)
    n.setSnInit(py_noise_config.sn_init)
    n.setSnMax(py_noise_config.sn_max)
    n.setThreshold(py_noise_config.threshold)
    return n

cdef prepare_result_item(ResultItem item):
    return Prediction(tuple(item.coords.as_vector()), item.val, item.pred_1sample, item.pred_avg, item.var)

cdef class TrainSession:
    """Class for doing a training run in smurff

    A simple use case could be:

    >>> session = smurff.TrainSession(burnin = 5, nsamples = 5)
    >>> session.addTrainAndTest(Ydense)
    >>> session.run()

        
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

    save_prefix: path
        Path where to store the samples. The path includes the directory name, as well
        as the initial part of the file names.

    save_freq: int
        - N>0: save every Nth sample
        - N==0: never save a sample
        - N==-1: save only the last sample

    save_extension: { ".csv", ".ddm" }
        - .csv: save in textual csv file format
        - .ddm: save in binary file format

    checkpoint_freq: int
        Save the state of the session every N seconds.

    csv_status: filepath
        Stores limited set of parameters, indicative for training progress in this file. See :class:`StatusItem`

    """
    cdef shared_ptr[ISession] ptr;
    cdef Config config
    cdef NoiseConfig noise_config
    cdef shared_ptr[StatusItem] status_item
    cdef readonly int nmodes
    cdef readonly int num_latent
    cdef readonly int verbose
    cdef vector[string] prior_types

    #
    # construction functions
    #
    def __init__(self,
        priors           = [ "normal", "normal" ],
        num_latent       = NUM_LATENT_DEFAULT_VALUE,
        num_threads      = NUM_THREADS_DEFAULT_VALUE,
        burnin           = BURNIN_DEFAULT_VALUE,
        nsamples         = NSAMPLES_DEFAULT_VALUE,
        seed             = RANDOM_SEED_DEFAULT_VALUE,
        threshold        = None,
        verbose          = 1,
        save_prefix      = None,
        save_extension   = None,
        save_freq        = None,
        checkpoint_freq  = None,
        csv_status       = None):

        self.num_latent = num_latent
        self.nmodes = len(priors)
        self.verbose = verbose

        if save_prefix is None and save_freq:
            save_prefix = tempfile.mkdtemp()

        if save_prefix and not os.path.isabs(save_prefix):
            save_prefix = os.path.join(os.getcwd(), save_prefix)
        
        if save_prefix:
            dir = os.path.dirname(save_prefix)
            if not os.path.exists(dir):
                os.makedirs(dir)

        prior_types = [ p.encode('UTF-8') for p in priors ]
        self.config.setPriorTypes(prior_types)
        self.config.setNumLatent(num_latent)
        self.config.setNumThreads(num_threads)
        self.config.setBurnin(burnin)
        self.config.setNSamples(nsamples)
        self.config.setVerbose(verbose)

        if seed:           self.config.setRandomSeed(seed)
        if threshold is not None:
                           self.config.setThreshold(threshold)
        if save_prefix:    self.config.setSavePrefix(save_prefix.encode('UTF-8'))
        if save_extension: self.config.setSaveExtension(save_extension.encode('UTF-8'))
        if save_freq:      self.config.setSaveFreq(save_freq)
        if checkpoint_freq:self.config.setCheckpointFreq(checkpoint_freq)

    def addTrainAndTest(self, Y, Ytest = None, noise = PyNoiseConfig(), is_scarce = True):
        """Adds a train and optionally a test matrix as input data to this TrainSession

        Parameters
        ----------

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Train matrix/tensor 
       
        Ytest : :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Test matrix/tensor. Mainly used for calculating RMSE.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        """
        self.noise_config = prepare_noise_config(noise)
        train, test = prepare_train_and_test(Y, Ytest, self.noise_config, is_scarce)
        self.config.setTrain(train)

        if Ytest is not None:
            self.config.setTest(test)

    def addSideInfo(self, mode, Y, noise = PyNoiseConfig(), tol = 1e-6, direct = False):
        """Adds fully known side info, for use in with the macau or macauone prior

        mode : int
            dimension to add side info (rows = 0, cols = 1)

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix
            Side info matrix/tensor 
            Y should have as many rows in Y as you have elemnts in the dimension selected using `mode`.
            Columns in Y are features for each element.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`
        
        direct : boolean
            - When True, uses a direct inversion method. 
            - When False, uses a CG solver 

            The direct method is only feasible for a small (< 100K) number of features.

        tol : float
            Tolerance for the CG solver.

        """
        self.noise_config = prepare_noise_config(noise)
        self.config.addSideInfoConfig(mode, prepare_sideinfo(Y, self.noise_config, tol, direct))

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
        self.noise_config = prepare_noise_config(PyNoiseConfig())
        if len(Lambda.shape) == 3:
            assert Lambda.shape[0] == self.num_latent
            assert Lambda.shape[1] == self.num_latent
            Lambda = Lambda.reshape(self.num_latent * self.num_latent, Lambda.shape[2], order='F')

        self.config.addPropagatedPosterior(
            mode,
            shared_ptr[MatrixConfig](prepare_dense_matrix(mu, self.noise_config)),
            shared_ptr[MatrixConfig](prepare_dense_matrix(Lambda, self.noise_config))
        )


    def addData(self, pos, Y, is_scarce = False, noise = PyNoiseConfig()):
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

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`
        
        """
        self.noise_config = prepare_noise_config(noise)
        self.config.addAuxData(prepare_auxdata(Y, pos, is_scarce, self.noise_config))

    cdef ISession* ptr_get(self) except *:
        if (not self.ptr.get()):
            raise ValueError("Session not initialized")
        return self.ptr.get()

    # 
    # running functions
    #

    def init(self):
        """Initializes the `TrainSession` after all data has been added.

        You need to call this method befor calling :meth:`step`, unless you call :meth:`run`

        Returns
        -------
        :class:`StatusItem` of the session.

        """

        self.ptr = SessionFactory.create_py_session_from_config(self.config)
        self.ptr_get().init()
        logging.info(self)
        return self.getStatus()

    def __dealloc__(self):
        if (self.ptr.get()):
            self.ptr.reset()

    def step(self):
        """Does on sampling or burnin iteration.

        Returns
        -------
        - When a step was executed: :class:`StatusItem` of the session.
        - After the last iteration, when no step was executed: `None`.

        """
        not_done = self.ptr_get().step()
        
        if self.ptr_get().interrupted():
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

    #
    # get state
    #

    def __str__(self):
        try:
            return self.ptr_get().infoAsString().decode('UTF-8')
        except ValueError:
            return "Uninitialized SMURFF Train Session (call .init())"


    def getStatus(self):
        """ Returns :class:`StatusItem` with current state of the session

        """
        if self.ptr_get().getStatus():
            self.status_item = self.ptr_get().getStatus()
            status =  PyStatusItem(
                self.status_item.get().phase,
                self.status_item.get().iter,
                self.status_item.get().phase_iter,
                self.status_item.get().model_norms,
                self.status_item.get().rmse_avg,
                self.status_item.get().rmse_1sample,
                self.status_item.get().train_rmse,
                self.status_item.get().auc_1sample,
                self.status_item.get().auc_avg,
                self.status_item.get().elapsed_iter,
                self.status_item.get().nnz_per_sec,
                self.status_item.get().samples_per_sec)

            logging.info(status)
            
            return status
        else:
            return None

    def getConfig(self):
        """Get this `TrainSession`'s configuration in ini-file format

        """
        config_filename = tempfile.mkstemp()[1]
        self.config.save(config_filename.encode('UTF-8'))
        
        with open(config_filename, 'r') as f:
            ini_string = "".join(f.readlines())

        os.unlink(config_filename)

        return ini_string

    def makePredictSession(self):
        """Makes a :class:`PredictSession` based on the model
           that as built in this `TrainSession`.

        """
        rf = self.ptr_get().getRootFile().get().getFullPath().decode('UTF-8')
        return PredictSession(rf)

    def getTestPredictions(self):
        """Get predictions for test matrix.

        Returns
        -------
        list 
            list of :class:`Prediction`

        """
        py_items = []

        if self.ptr_get().getResultItems().size():
            cpp_items = self.ptr_get().getResultItems()
            it = cpp_items.begin()
            while it != cpp_items.end():
                py_items.append(prepare_result_item(deref(it)))
                inc(it)

        return py_items
    
    def getRmseAvg(self): 
        """Average RMSE across all samples for the test matrix

        """
        return self.ptr_get().getRmseAvg()
