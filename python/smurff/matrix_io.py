import numpy as np
import scipy.sparse
import scipy.io as sio
import os

def read_dense_float64(filename, order = 'F'):
    with open(filename) as f:
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        vals = np.fromfile(f, dtype=np.float64, count=nrow*ncol)
        return np.ndarray((nrow,ncol), buffer=vals, dtype=np.float64, order = order)

def read_sparse_float64(filename):
    with open(filename) as f:
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        nnz  = np.fromfile(f, dtype=np.int64, count=1)[0]
        rows = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        cols = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        vals = np.fromfile(f, dtype=np.float64, count=nnz)
        return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=[nrow, ncol])

def read_sparse_binary_matrix(filename):
    with open(filename) as f:
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        nnz  = np.fromfile(f, dtype=np.int64, count=1)[0]
        rows = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        cols = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        return scipy.sparse.coo_matrix((np.ones(nnz), (rows, cols)), shape=[nrow, ncol])

def write_dense_float64(filename, Y, order = 'F'):
    with open(filename, 'wb') as f:
        np.array(Y.shape[0]).astype(np.int64).tofile(f)
        np.array(Y.shape[1]).astype(np.int64).tofile(f)
        f.write(Y.astype(np.float64).tobytes(order = order))

def write_sparse_float64(filename, Y):
    with open(filename, 'wb') as f:
        Y = Y.tocoo(copy = False)
        np.array(Y.shape[0]).astype(np.int64).tofile(f)
        np.array(Y.shape[1]).astype(np.int64).tofile(f)
        np.array(Y.nnz).astype(np.int64).tofile(f)
        (Y.row + 1).astype(np.int32, copy=False).tofile(f)
        (Y.col + 1).astype(np.int32, copy=False).tofile(f)
        Y.data.astype(np.float64, copy=False).tofile(f)

def write_sparse_binary_matrix(filename, Y):
    with open(filename, 'wb') as f:
        Y = Y.tocoo(copy = False)
        np.array( Y.shape[0] ).astype(np.int64).tofile(f)
        np.array( Y.shape[1] ).astype(np.int64).tofile(f)
        np.array( Y.nnz ).astype(np.int64).tofile(f)
        (Y.row + 1).astype(np.int32, copy=False).tofile(f)
        (Y.col + 1).astype(np.int32, copy=False).tofile(f)

def read_dense_float64_tensor_as_matrix(filename, order = 'F'):
    with open(filename) as f:
        ndim = np.fromfile(f, dtype=np.int64, count=1)[0]
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        vals = np.fromfile(f, dtype=np.float64, count=nrow*ncol)
        return np.ndarray((nrow,ncol), buffer=vals, dtype=np.float64, order = order)

def write_dense_float64_matrix_as_tensor(filename, Y, order = 'F'):
    with open(filename, 'wb') as f:
        np.array(len(Y.shape)).astype(np.int64).tofile(f)
        np.array(Y.shape[0]).astype(np.int64).tofile(f)
        np.array(Y.shape[1]).astype(np.int64).tofile(f)
        f.write(Y.astype(np.float64).tobytes(order = order))

def read_sparse_float64_tensor_as_matrix(filename):
    with open(filename) as f:
        ndim = np.fromfile(f, dtype=np.int64, count=1)[0]
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        nnz  = np.fromfile(f, dtype=np.int64, count=1)[0]
        rows = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        cols = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        vals = np.fromfile(f, dtype=np.float64, count=nnz)
        return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=[nrow, ncol])

def write_sparse_float64_matrix_as_tensor(filename, Y):
    with open(filename, 'w') as f:
        Y = Y.tocoo(copy=False)
        np.array(len(Y.shape)).astype(np.uint64).tofile(f)
        np.array(Y.shape[0]).astype(np.uint64).tofile(f)
        np.array(Y.shape[1]).astype(np.uint64).tofile(f)
        np.array(Y.nnz).astype(np.uint64).tofile(f)
        (Y.row + 1).astype(np.uint32, copy=False).tofile(f)
        (Y.col + 1).astype(np.uint32, copy=False).tofile(f)
        Y.data.astype(np.float64, copy=False).tofile(f)

def my_mmwrite(filename, Y):
    with open(filename, 'wb') as f:
        sio.mmwrite(f, Y, symmetry='general')


def read_csv(filename):
    with open(filename, 'r') as f:
        nrow = int(f.readline())
        ncol = int(f.readline())
        Y = np.loadtxt(f, delimiter=',')
        assert(Y.shape == (nrow,ncol))

    return Y

def write_csv(filename, Y):
    with open(filename, 'wb') as f:
        header = "%d\n%d" % Y.shape
        np.savetxt(f, Y, header=header, comments = "", delimiter=',')


ext_map = {
        ".mtx": ( sio.mmread,                my_mmwrite ),
        ".mm":  ( sio.mmread,                my_mmwrite ),
        ".sbm": ( read_sparse_binary_matrix, write_sparse_binary_matrix ),
        ".sdm": ( read_sparse_float64,       write_sparse_float64 ),
        ".ddm": ( read_dense_float64,        write_dense_float64 ),
        ".csv": ( read_csv,                  write_csv ),
}

def read_matrix(filename, **kwargs):
    base, ext =  os.path.splitext(filename)
    return ext_map[ext][0](filename, **kwargs)

def write_matrix(filename, Y, **kwargs):
    base, ext =  os.path.splitext(filename)
    return ext_map[ext][1](filename, Y, **kwargs)


## example

## example

## example

# get the files from:
#
# http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
# http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm
#
# import macau
# import scipy.io
#
# ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
# ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")
#
# import making_input_for_macau_mpi
# making_input_for_macau_mpi.write_sparse_binary_matrix("chembl-IC50-compound-feat.sbm", ecfp)
# making_input_for_macau_mpi.write_sparse_float64("chembl-IC50-346targets.sdm", ic50)
