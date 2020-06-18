import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers

from .helper import SparseTensor

def make_train_test(Y, ntest, shape = None):
    """Splits a sparse matrix Y into a train and a test matrix.

    Parameters
    ----------
        Y : scipy sparse matrix (coo_matrix, csr_matrix or csc_matrix)
             or
            numpy dense ndarray 
             or
            pandas DataFrame or smurff.SparseTensor

            Matrix/Array/Tensor to split

        ntest : float <1.0 or integer.
           - if float, then indicates the ratio of test cells
           - if integer, then indicates the number of test cells

    Returns
    -------
        Ytrain : coo_matrix
            train part

        Ytest : coo_matrix
            test part
    """
    if isinstance(Y, pd.DataFrame) or isinstance(Y, SparseTensor):
        return make_train_test_df(Y, ntest, shape)

    if isinstance(Y, np.ndarray):
        _, Ytest = make_train_test(sp.sparse.coo_matrix(Y), ntest, shape)
        return Y, Ytest
    
    if not sp.sparse.issparse(Y):
        raise TypeError("Unsupported Y type: " + str(type(Y)))

    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")

    Y = Y.tocoo(copy = False)
    if ntest < 1:
        ntest = Y.nnz * ntest
    ntest = int(round(ntest))
    rperm = np.random.permutation(Y.nnz)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    if shape is None:
        shape = Y.shape

    Ytrain = sp.sparse.coo_matrix( (Y.data[train], (Y.row[train], Y.col[train])), shape=shape )
    Ytest  = sp.sparse.coo_matrix( (Y.data[test],  (Y.row[test],  Y.col[test])),  shape=shape )
    return Ytrain, Ytest

def make_train_test_df(Y, ntest, shape = None):
    """Splits rows of dataframe Y into a train and a test dataframe.
       Y      pandas dataframe
       ntest  either a float below 1.0 or integer.
              if float, then indicates the ratio of test cells
              if integer, then indicates the number of test cells
       returns Ytrain, Ytest (type coo_matrix)
    """
    if type(Y) != pd.core.frame.DataFrame:
        raise TypeError("Y should be DataFrame.")
    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")

    ## randomly spliting train-test
    if ntest < 1:
        ntest = Y.shape[0] * ntest

    ntest  = int(round(ntest))
    rperm  = np.random.permutation(Y.shape[0])
    train  = rperm[ntest:]
    test   = rperm[0:ntest]

    Ytrain = SparseTensor(Y.iloc[train], shape)
    Ytest = SparseTensor(Y.iloc[test], Ytrain.shape)

    return Ytrain, Ytest
