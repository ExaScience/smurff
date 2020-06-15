#!/usr/bin/env python

import unittest
import numpy as np
import scipy as sp
from smurff.center import mean, center, std, scale, center_and_scale

class TestCentering(unittest.TestCase):
    def test_rows_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = mean(origin_matrix, 'rows')
        centered_matrix = center(origin_matrix, 'rows', origin_matrix_mean)
        expected_matrix = np.array([[-1.5, -0.5, 0.5, 1.5],
                                    [-1.5, -0.5, 0.5, 1.5],
                                    [-1.5, -0.5, 0.5, 1.5]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_cols_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = mean(origin_matrix, 'cols')
        centered_matrix = center(origin_matrix, 'cols', origin_matrix_mean)
        expected_matrix = np.array([[-4,-4, -4, -4],
                                    [ 0, 0,  0,  0],
                                    [ 4, 4,  4,  4]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_global_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = mean(origin_matrix, 'global')
        centered_matrix = center(origin_matrix, 'global', origin_matrix_mean)
        expected_matrix = np.array([[-5.5, -4.5, -3.5, -2.5],
                                    [-1.5, -0.5,  0.5,  1.5],
                                    [ 2.5,  3.5,  4.5,  5.5]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_none_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = mean(origin_matrix, 'none')
        centered_matrix = center(origin_matrix, 'none', origin_matrix_mean)
        expected_matrix = np.array([[1,  2,  3,  4],
                                    [5,  6,  7,  8],
                                    [9, 10, 11, 12]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_dense_matrix_rows_scaling(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_std = std(origin_matrix, 'rows')
        expected_matrix_std = [[1.11803399, 1.11803399, 1.11803399]]
        self.assertTrue(np.allclose(origin_matrix_std, expected_matrix_std))

        scaled_matrix = scale(origin_matrix, 'rows', origin_matrix_std)
        expected_matrix = [[ 0.89442719, 1.78885438, 2.68328157,  3.57770876],
                           [ 4.47213595, 5.36656315, 6.26099034,  7.15541753],
                           [ 8.04984472, 8.94427191, 9.83869910, 10.73312629]]
        self.assertTrue(np.allclose(scaled_matrix, expected_matrix))

    def test_sparse_matrix_rows_scaling(self):
        origin_matrix_rows = np.array([0, 0, 1, 1, 2,  2])
        origin_matrix_cols = np.array([0, 1, 0, 1, 0,  1])
        origin_matrix_vals = np.array([1, 2, 5, 6, 9, 10], dtype=sp.float64)
        origin_matrix = sp.sparse.coo_matrix((origin_matrix_vals, (origin_matrix_rows, origin_matrix_cols)), shape=(3, 4))

        origin_matrix_std = std(origin_matrix, 'rows')
        expected_matrix_std = [[1.29099445, 4.50924975, 7.76745347]]
        self.assertTrue(np.allclose(origin_matrix_std, expected_matrix_std))

        scaled_matrix = scale(origin_matrix, 'rows', origin_matrix_std)
        expected_matrix_rows = np.array([0, 1, 2, 0, 1, 2])
        expected_matrix_cols = np.array([0, 0, 0, 1, 1, 1])
        expected_matrix_vals  = [ 0.7745966692414834
                                , 1.1088319064318592
                                , 1.1586809036417614
                                , 1.5491933384829668
                                , 1.3305982877182312
                                , 1.2874232262686236
                                ]
        expected_matrix = sp.sparse.coo_matrix((expected_matrix_vals, (expected_matrix_rows, expected_matrix_cols)), shape=(3, 4))
        self.assertTrue(np.allclose(scaled_matrix.todense(), expected_matrix.todense()))

    def test_dense_matrix_cols_scaling(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_std = std(origin_matrix, 'cols')
        expected_matrix_std = [[3.26598632, 3.26598632, 3.26598632, 3.26598632]]
        self.assertTrue(np.allclose(origin_matrix_std, expected_matrix_std))

        scaled_matrix = scale(origin_matrix, 'cols', origin_matrix_std)
        expected_matrix = [[0.30618622, 0.61237244, 0.91855865, 1.22474487],
                           [1.53093109, 1.83711731, 2.14330352, 2.44948974],
                           [2.75567596, 3.06186218, 3.36804840, 3.67423461]]
        self.assertTrue(np.allclose(scaled_matrix, expected_matrix))

    def test_sparse_matrix_cols_scaling(self):
        origin_matrix_rows = np.array([0, 0, 1, 1, 2,  2])
        origin_matrix_cols = np.array([0, 1, 2, 3, 0,  1])
        origin_matrix_vals = np.array([1, 2, 7, 8, 9, 10], dtype=sp.float64)
        origin_matrix = sp.sparse.coo_matrix((origin_matrix_vals, (origin_matrix_rows, origin_matrix_cols)), shape=(3, 4))

        origin_matrix_std = std(origin_matrix, 'cols')
        expected_matrix_std = [[4.35889894, 4.76095229, 5.71547607, 6.53197265]]
        self.assertTrue(np.allclose(origin_matrix_std, expected_matrix_std))

        scaled_matrix = scale(origin_matrix, 'cols', origin_matrix_std)
        expected_matrix_rows = np.array([0, 0, 1, 1, 2,  2])
        expected_matrix_cols = np.array([0, 1, 2, 3, 0,  1])
        expected_matrix_vals = [ 0.22941573387056174
                               , 0.42008402520840293
                               , 1.22474487139158920
                               , 1.22474487139158900
                               , 2.06474160483505550
                               , 2.10042012604201500
                               ]
        expected_matrix = sp.sparse.coo_matrix((expected_matrix_vals, (expected_matrix_rows, expected_matrix_cols)), shape=(3, 4))
        self.assertTrue(np.allclose(scaled_matrix.todense(), expected_matrix.todense()))

    def test_dense_matrix_global_scaling(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_std = std(origin_matrix, 'global')
        expected_matrix_std = 3.45205252953
        self.assertTrue(np.allclose(origin_matrix_std, expected_matrix_std))

        scaled_matrix = scale(origin_matrix, 'global', origin_matrix_std)
        expected_matrix = [[0.28968273, 0.57936546, 0.86904819, 1.15873092],
                           [1.44841365, 1.73809638, 2.02777911, 2.31746184],
                           [2.60714457, 2.89682730, 3.18651003, 3.47619276]]
        self.assertTrue(np.allclose(scaled_matrix, expected_matrix))

    def test_sparse_matrix_global_scaling(self):
        origin_matrix_rows = np.array([0, 0, 1, 1, 2,  2])
        origin_matrix_cols = np.array([0, 1, 2, 3, 0,  1])
        origin_matrix_vals = np.array([1, 2, 7, 8, 9, 10], dtype=sp.float64)
        origin_matrix = sp.sparse.coo_matrix((origin_matrix_vals, (origin_matrix_rows, origin_matrix_cols)), shape=(3, 4))

        origin_matrix_std = std(origin_matrix, 'global')
        expected_matrix_std = 3.43592135468
        self.assertTrue(np.allclose(origin_matrix_std, expected_matrix_std))

        scaled_matrix = scale(origin_matrix, 'global', origin_matrix_std)
        expected_matrix_rows = np.array([0, 0, 1, 1, 2,  2])
        expected_matrix_cols = np.array([0, 1, 2, 3, 0,  1])
        expected_matrix_vals = [ 0.2910427500435996
                               , 0.5820855000871992
                               , 2.0372992503051970
                               , 2.3283420003487967
                               , 2.6193847503923960
                               , 2.9104275004359956
                               ]
        expected_matrix = sp.sparse.coo_matrix((expected_matrix_vals, (expected_matrix_rows, expected_matrix_cols)), shape=(3, 4))
        self.assertTrue(np.allclose(scaled_matrix.todense(), expected_matrix.todense()))

    def test_dense_matrix_none_scaling(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_std = std(origin_matrix, 'none')
        scaled_matrix = scale(origin_matrix, 'none', origin_matrix_std)
        self.assertTrue(np.allclose(scaled_matrix, origin_matrix))

    def test_sparse_matrix_none_scaling(self):
        origin_matrix_rows = np.array([0, 0, 1, 1, 2,  2])
        origin_matrix_cols = np.array([0, 1, 2, 3, 0,  1])
        origin_matrix_vals = np.array([1, 2, 7, 8, 9, 10], dtype=sp.float64)
        origin_matrix = sp.sparse.coo_matrix((origin_matrix_vals, (origin_matrix_rows, origin_matrix_cols)), shape=(3, 4))
        origin_matrix_std = std(origin_matrix, 'none')
        scaled_matrix = scale(origin_matrix, 'none', origin_matrix_std)
        self.assertTrue(np.allclose(scaled_matrix.todense(), origin_matrix.todense()))

    def test_dense_matrix_rows_center_and_scale(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'rows')
        expected_matrix = [[-1.34164079, -0.4472136, 0.4472136, 1.34164079],
                           [-1.34164079, -0.4472136, 0.4472136, 1.34164079],
                           [-1.34164079, -0.4472136, 0.4472136, 1.34164079]]
        expected_matrix_mean = [2.5, 6.5, 10.5]
        expected_matrix_std = [1.11803399, 1.11803399, 1.11803399]
        self.assertTrue(np.allclose(centered_and_scaled_matrix, expected_matrix))
        self.assertTrue(np.allclose(centered_and_scaled_matrix_mean, expected_matrix_mean))
        self.assertTrue(np.allclose(centered_and_scaled_matrix_std, expected_matrix_std))

    def test_dense_matrix_rows_center_and_scale_with_mean_only(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'rows', with_std=False)
        expected_matrix = np.array([[-1.5, -0.5, 0.5, 1.5],
                                    [-1.5, -0.5, 0.5, 1.5],
                                    [-1.5, -0.5, 0.5, 1.5]])
        self.assertTrue(np.allclose(centered_and_scaled_matrix, expected_matrix))
        self.assertEqual(centered_and_scaled_matrix_std, None)

    def test_dense_matrix_rows_center_and_scale_with_std_only(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'rows', with_mean=False)
        expected_matrix = [[ 0.89442719, 1.78885438, 2.68328157,  3.57770876],
                           [ 4.47213595, 5.36656315, 6.26099034,  7.15541753],
                           [ 8.04984472, 8.94427191, 9.83869910, 10.73312629]]
        self.assertTrue(np.allclose(centered_and_scaled_matrix, expected_matrix))
        self.assertEqual(centered_and_scaled_matrix_mean, None)

    def test_dense_matrix_rows_center_and_scale_with_no_mean_and_std(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'rows', with_mean=False, with_std=False)
        self.assertTrue(np.allclose(centered_and_scaled_matrix, origin_matrix))
        self.assertEqual(centered_and_scaled_matrix_mean, None)
        self.assertEqual(centered_and_scaled_matrix_std, None)

    def test_dense_matrix_cols_center_and_scale(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'cols')
        expected_matrix = [[-1.22474487, -1.22474487, -1.22474487, -1.22474487],
                           [ 0.,          0.,          0.,          0.        ],
                           [ 1.22474487,  1.22474487,  1.22474487,  1.22474487]]
        expected_matrix_mean = [5., 6., 7., 8.]
        expected_matrix_std = [3.26598632, 3.26598632, 3.26598632, 3.26598632]
        self.assertTrue(np.allclose(centered_and_scaled_matrix, expected_matrix))
        self.assertTrue(np.allclose(centered_and_scaled_matrix_mean, expected_matrix_mean))
        self.assertTrue(np.allclose(centered_and_scaled_matrix_std, expected_matrix_std))

    def test_dense_matrix_global_center_and_scale(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'global')
        expected_matrix = [[-1.59325501, -1.30357228, -1.01388955, -0.72420682],
                           [-0.43452409, -0.14484136,  0.14484136,  0.43452409],
                           [ 0.72420682,  1.01388955,  1.30357228,  1.59325501]]
        expected_matrix_mean = 6.5
        expected_matrix_std = 3.45205252953
        self.assertTrue(np.allclose(centered_and_scaled_matrix, expected_matrix))
        self.assertTrue(np.allclose(centered_and_scaled_matrix_mean, expected_matrix_mean))
        self.assertTrue(np.allclose(centered_and_scaled_matrix_std, expected_matrix_std))

    def test_dense_matrix_none_center_and_scale(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        centered_and_scaled_matrix, centered_and_scaled_matrix_mean, centered_and_scaled_matrix_std = center_and_scale(origin_matrix, 'none')
        self.assertTrue(np.allclose(centered_and_scaled_matrix, origin_matrix))
        self.assertEqual(centered_and_scaled_matrix_mean, None)
        self.assertEqual(centered_and_scaled_matrix_std, None)

if __name__ == '__main__':
    unittest.main()
