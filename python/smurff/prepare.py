import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers

from .helper import SparseTensor

def make_train_test(Y, ntest, shape = None, seed = None):
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
        Ytrain : csr_matrix
            train part

        Ytest : csr_matrix
            test part
    """
    if isinstance(Y, pd.DataFrame) or isinstance(Y, SparseTensor):
        return make_train_test_df(Y, ntest, shape)

    if isinstance(Y, np.ndarray):
        nmodes = len(Y.shape)
        if (nmodes > 2):
            Ysparse = SparseTensor(Y)
        else:
            Ysparse = sp.sparse.coo_matrix(Y)

        return make_train_test(Ysparse, ntest, shape)
    
    if not sp.sparse.issparse(Y):
        raise TypeError("Unsupported Y type: " + str(type(Y)))

    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")

    Y = Y.tocoo(copy = False)
    if ntest < 1:
        ntest = Y.nnz * ntest
    ntest = int(round(ntest))
    ntest = max(1,ntest)

    np.random.seed(seed)
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
       returns Ytrain, Ytest (type SparseTensor)
    """
    if  isinstance(Y, pd.DataFrame):
        return make_train_test_df(SparseTensor(Y), ntest, Y.shape)

    if not isinstance(Y, SparseTensor):
        raise TypeError("Y should be DataFrame or SparseTensor.")

    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")

    ## randomly spliting train-test
    if ntest < 1:
        ntest = Y.nnz * ntest

    ntest = int(round(ntest))
    ntest = max(1,ntest)

    rperm  = np.random.permutation(Y.nnz)
    train  = rperm[ntest:]
    test   = rperm[0:ntest]

    Ytrain = SparseTensor(
        ( Y.values[train], [ idx[train] for idx in Y.columns ] ),
        Y.shape)

    Ytest  = SparseTensor(
        ( Y.values[test], [ idx[test] for idx in Y.columns ] ),
        Y.shape)

    return Ytrain, Ytest
