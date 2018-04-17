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

cimport numpy as np
import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers
import tempfile
import os

from .helper import SparseTensor, PyNoiseConfig
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

cdef MatrixConfig* prepare_sparse_matrix(X, NoiseConfig noise_config, is_scarse) except +:
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

    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), rows_vector_shared_ptr, cols_vector_shared_ptr, vals_vector_shared_ptr, noise_config, is_scarse)
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
    raise NotImplementedError()

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

cdef (shared_ptr[TensorConfig], shared_ptr[TensorConfig]) prepare_train_and_test(train, test, NoiseConfig noise_config) except +:

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
        if isinstance(train, MATRIX_TYPES) and (not isinstance(test, MATRIX_TYPES)):
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
        train_config = prepare_sparse_matrix(train, noise_config, True)
    elif isinstance(train, DENSE_TENSOR_TYPES) and len(train.shape) > 2:
        train_config = prepare_dense_tensor(train, noise_config)
    elif isinstance(train, SPARSE_TENSOR_TYPES):
        train_config = prepare_sparse_tensor(train, noise_config, True)
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
    return Prediction(tuple(item.coords.as_vector()), item.val, item.pred_1sample, item.pred_avg , item.var)

class StepItem:
    pass

cdef class TrainSession:
    cdef shared_ptr[ISession] ptr;
    cdef Config config
    cdef NoiseConfig noise_config
    cdef readonly int nmodes

    def __init__(self,
        priors           = [ "normal", "normal" ],
        num_latent       = NUM_LATENT_DEFAULT_VALUE,
        burnin           = BURNIN_DEFAULT_VALUE,
        nsamples         = NSAMPLES_DEFAULT_VALUE,
        seed             = RANDOM_SEED_DEFAULT_VALUE,
        verbose          = VERBOSE_DEFAULT_VALUE,
        save_prefix      = None,
        save_extension   = None,
        save_freq        = None,
        csv_status       = None):

        self.nmodes = len(priors)

        if save_prefix is None and save_freq:
            save_prefix = os.path.join(tempfile.mkdtemp(), "save")

        for p in priors: self.config.addPriorType(p.encode('UTF-8'))
        self.config.setNumLatent(num_latent)
        self.config.setBurnin(burnin)
        self.config.setNSamples(nsamples)
        self.config.setVerbose(verbose)

        if seed:           self.config.setRandomSeed(seed)
        if save_prefix:    self.config.setSavePrefix(save_prefix.encode('UTF-8'))
        if save_extension: self.config.setSaveExtension(save_extension.encode('UTF-8'))
        if save_freq:      self.config.setSaveFreq(save_freq)
        if csv_status:     self.config.setCsvStatus(csv_status.encode('UTF-8'))

    def addTrainAndTest(self, Y, Ytest = None, noise = PyNoiseConfig()):
        self.noise_config = prepare_noise_config(noise)
        train, test = prepare_train_and_test(Y, Ytest, self.noise_config)
        self.config.setTrain(train)

        if Ytest is not None:
            self.config.setTest(test)

    def addSideInfo(self, mode, Y, noise = PyNoiseConfig(), tol = 1e-6, direct = False):
        self.noise_config = prepare_noise_config(noise)
        self.config.addSideInfoConfig(mode, prepare_sideinfo(Y, self.noise_config, tol, direct))

    def addData(self, pos, ad, is_scarce = False, noise = PyNoiseConfig()):
        self.noise_config = prepare_noise_config(noise)
        self.config.addAuxData(prepare_auxdata(ad, pos, is_scarce, self.noise_config))

    def init(self):
        self.ptr = SessionFactory.create_py_session_from_config(self.config)
        return self.ptr.get().init()

    def step(self):
        not_done = self.ptr.get().step()
        if self.ptr.get().interrupted():
            raise KeyboardInterrupt
        return not_done

    def getCurrentSample(self):
        not_done = self.ptr.get().step()

    def getConfig(self):
        config_filename = tempfile.mkstemp()[1]
        self.config.save(config_filename.encode('UTF-8'))
        
        with open(config_filename, 'r') as f:
            ini_string = "".join(f.readlines())

        os.unlink(config_filename)

        return ini_string

    def makePredictSession(self):
        rf = self.ptr.get().getRootFile().get().getRootFileName()
        return PredictSession.fromRootFile(rf)

    def getTestPredictions(self):
        """ Create Python list of Prediction from C++ vector of Predictions """
        py_items = []

        cpp_items = self.ptr.get().getResult().get()
        it = cpp_items.begin()
        while it != cpp_items.end():
            py_items.append(prepare_result_item(deref(it)))
            inc(it)

        return py_items
    
    def getRmseAvg(self): 
        return self.ptr.get().getRmseAvg()
