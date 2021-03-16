import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers

from .helper import SparseTensor

def make_sparse(Y, nnz, shape = None, seed = None):
    Ytr, Yte = make_train_test(Y, nnz, shape, seed)
    return Yte

def make_train_test(Y, ntest, shape = None, seed = None):
    """Splits a sparse matrix Y into a train and a test matrix.

    Parameters
    ----------
        Y : :class:`scipy.spmatrix`, (coo_matrix, csr_matrix or csc_matrix) or
            :class:`numpy.ndarray` or
            :class:`pandas.DataFrame` or
            :class:`smurff.SparseTensor`

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
    if  isinstance(Y, pd.DataFrame):
        return make_train_test(SparseTensor(Y), ntest, Y.shape, seed)        

    if isinstance(Y, np.ndarray):
        nmodes = len(Y.shape)
        if (nmodes > 2):
            Ysparse = SparseTensor(Y)
        else:
            Ysparse = sp.sparse.coo_matrix(Y)

        return make_train_test(Ysparse, ntest, shape, seed)
    
    if sp.sparse.issparse(Y):
        Y = Y.tocoo(copy = False)
    elif not isinstance(Y, SparseTensor):
        raise TypeError("Unsupported Y type: " + str(type(Y)))

    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")

    if ntest < 1:
        ntest = Y.nnz * ntest
    ntest = int(round(ntest))
    ntest = max(1,ntest)

    if seed is not None:
        np.random.seed(seed)

    rperm = np.random.permutation(Y.nnz)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    if shape is None:
        shape = Y.shape

    if sp.sparse.issparse(Y):
        Ytrain = sp.sparse.coo_matrix( (Y.data[train], (Y.row[train], Y.col[train])), shape=shape )
        Ytest  = sp.sparse.coo_matrix( (Y.data[test],  (Y.row[test],  Y.col[test])),  shape=shape )
    else:
        assert isinstance(Y, SparseTensor)

        Ytrain = SparseTensor(
            ( Y.values[train], [ idx[train] for idx in Y.columns ] ),
            Y.shape)

        Ytest  = SparseTensor(
            ( Y.values[test], [ idx[test] for idx in Y.columns ] ),
            Y.shape)

    return Ytrain, Ytest