import pandas as pd
import numpy as np
import scipy.sparse as sp
import math
import tempfile
import os

from .wrapper import NoiseConfig
from .result import Prediction
from . import wrapper

def temp_savename():
    return os.path.join(tempfile.mkdtemp(), "smurff_temp_output.h5")


class SparseTensor(wrapper.SparseTensor):
    """Wrapper around a pandas DataFrame to represent a sparse tensor

       The DataFrame should have N index columns (int type) and 1 value column (float type)
       N is the dimensionality of the tensor

       You can also specify the shape of the tensor. If you don't it is detected automatically.
    """
       
    def __init__(self, data, shape = None):
        if isinstance(data, tuple):
            values, columns = data

            if shape is None:
               shape = [ c.max() + 1 for c in columns ]
        elif isinstance(data, SparseTensor):
            columns = data.columns
            values = data.values

            if shape is None:
                shape = data.shape
        elif isinstance(data, pd.DataFrame):
            idx_column_names = list(filter(lambda c: data[c].dtype==np.int64 or data[c].dtype==np.int32, data.columns))
            val_column_names = list(filter(lambda c: data[c].dtype==np.float32 or data[c].dtype==np.float64, data.columns))

            if len(val_column_names) != 1:
                error_msg = "tensor has {} float columns but must have exactly 1 value column.".format(len(val_column_names))
                raise ValueError(error_msg)

            columns = [ data[c].values for c in idx_column_names ]
            values = data[val_column_names[0]].values

            if shape is None:
                shape = [ c.max() + 1 for c in columns ]

        elif isinstance(data, np.ndarray):
            if shape is None:
                shape = data.shape
            columns = [ col.flatten() for col in np.indices(shape) ]
            values = data.flatten()
        else:
            error_msg = "Unsupported sparse tensor data type: {}".format(data)
            raise ValueError(error_msg)

        for col in columns:
            assert len(col) == len(values), "Unequal column lenghts:\n%s\n%s" % (col, values)

        super().__init__(shape,columns,values)

    def __str__(self):
        return "SparseTensor(shape = " + str(self.shape) + ", nnz: " + str(len(self.values)) + "): \n" + \
            "\n".join( [ "(" + ",".join([str(c[i]) for  c in self.columns ]) + "): " + str(v) for i, v in enumerate(self.values) ] ) +  \
        ")"

    def coords(self, i):
        return tuple([ self.columns[d][i] for d in range(self.ndim) ])

    def toResult(self):
        return [ Prediction(self.coords(i), val) for i, val in enumerate(self.values) ]
    
    def toMatrix(self):
        assert self.ndim == 2
        return scipy.sparse.coo_matrix( (self.values, (self.columns[0], self.columns[1]) ) ) 

def find(Y):
    if isinstance(Y, sp.spmatrix):
        I, J, V = sp.find(Y)
        return (I, J), V
    
    if isinstance(Y, wrapper.SparseTensor):
        return Y.columns, Y.values

    raise NotImplemented

class FixedNoise(NoiseConfig):
    def __init__(self, precision = 5.0): 
        NoiseConfig.__init__(self, "fixed", precision)

class SampledNoise(NoiseConfig):
    def __init__(self, precision = 5.0): 
        NoiseConfig.__init__(self, "sampled", precision)

class AdaptiveNoise(NoiseConfig):
    def __init__(self, sn_init = 5.0, sn_max = 10.0): 
        NoiseConfig.__init__(self, "adaptive", sn_init = sn_init, sn_max = sn_max)

class ProbitNoise(NoiseConfig):
    def __init__(self, threshold = 0.): 
        NoiseConfig.__init__(self, "probit", threshold = threshold)