#!/usr/bin/env python

import numpy as np
from .prepare import make_sparse

def gen_matrix(shape, num_latent, density = 1.0 ):
    X = np.random.normal(size=(shape[0],num_latent))
    W = np.random.normal(size=(shape[1],num_latent))
    Y = np.dot(X, W.transpose()) + np.random.normal(size=shape)
    Y = make_sparse(Y, density)
    return Y, X ,W

def gen_tensor(shape, num_latent, density = 1.0 ):
    X = [ np.random.normal(size=(s,num_latent)) for s in shape ]
    # einsum ( [X[0], (1,0), X[1], (2,0), ... ] )
    Y = np.einsum(*sum([ [x, (i+1,0)] for i,x in enumerate(X) ], []))
    if density < 1.0:
        Y = make_sparse(Y, Y.size * density)
    return Y, X