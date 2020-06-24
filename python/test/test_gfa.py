import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

verbose = 0

class TestGFA(unittest.TestCase):

    # Python 2.7 @unittest.skip fix
    __name__ = "TestSmurff"

    def test_gfa_1view(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        predictions = smurff.gfa([Y], Ytest=Ytest, num_latent=4, verbose=verbose, burnin=5, nsamples=5)
        self.assertEqual(Ytest.nnz, len(predictions))

    def test_gfa_2view(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        predictions = smurff.gfa([Y, Y], Ytest=Ytest, num_latent=4, verbose=verbose, burnin=5, nsamples=5)
        self.assertEqual(Ytest.nnz, len(predictions))

    def test_gfa_3view(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        predictions = smurff.gfa([Y, Y, Y], Ytest=Ytest, num_latent=4, verbose=verbose, burnin=5, nsamples=5)
        self.assertEqual(Ytest.nnz, len(predictions))

    def test_gfa_mixedview(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        D1 = np.random.randn(10, 2)
        D2 = scipy.sparse.rand(10, 5, 0.2)
        predictions = smurff.gfa([Y, D1, D2], Ytest=Ytest, num_latent=4, verbose=verbose, burnin=5, nsamples=5)
        self.assertEqual(Ytest.nnz, len(predictions))


if __name__ == '__main__':
    unittest.main()
