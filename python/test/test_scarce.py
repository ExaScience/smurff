#!/usr/bin/env python

import unittest
import numpy as np
import scipy.sparse as sp
import smurff


def matrix_with_explicit_zeros():
    matrix_rows = np.array([0, 0, 1, 1, 2,  2])
    matrix_cols = np.array([0, 1, 0, 1, 0,  1])
    matrix_vals = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
    matrix = sp.coo_matrix((matrix_vals, (matrix_rows, matrix_cols)), shape=(3, 4))
    return matrix

class TestScarce(unittest.TestCase):
    """
    We make sure that we do not eliminate zeros in SMURFF
    accidentally
    """

    def test_simple(self):
        matrix = matrix_with_explicit_zeros()
        self.assertTrue(matrix.nnz == 6)
        
        matrix.eliminate_zeros()
        self.assertTrue(matrix.nnz == 3)

    def test_smurff(self):
        matrix = matrix_with_explicit_zeros()
        self.assertTrue(matrix.nnz == 6)

        predictions = smurff.bpmf(matrix, Ytest=matrix, num_latent=4, burnin=5, nsamples=5)
        self.assertEqual(len(predictions), 6)
        
if __name__ == '__main__':
    unittest.main()
