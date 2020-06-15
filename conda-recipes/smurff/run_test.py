import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

verbose = 0

class TestBPMF(unittest.TestCase):

    # Python 2.7 @unittest.skip fix
    __name__ = "TestSmurff"

    def test_bpmf(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        predictions = smurff.bpmf(Y,
                                Ytest=Ytest,
                                num_latent=4,
                                verbose=verbose,
                                burnin=50,
                                nsamples=50)
        self.assertEqual(Ytest.nnz, len(predictions))

    def test_bpmf_numerictest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = 0.3
        X, Xt = smurff.make_train_test(X, Xt)
        smurff.bpmf(X,
                      Ytest=Xt,
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=verbose)

    def test_bpmf_emptytest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        smurff.bpmf(X,
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=verbose)

    def test_bpmf_tensor(self):
        np.random.seed(1234)
        Y = smurff.SparseTensor(pd.DataFrame({
            "A": np.random.randint(0, 5, 7),
            "B": np.random.randint(0, 4, 7),
            "C": np.random.randint(0, 3, 7),
            "value": np.random.randn(7)
        }))
        Ytest = smurff.SparseTensor(pd.DataFrame({
            "A": np.random.randint(0, 5, 5),
            "B": np.random.randint(0, 4, 5),
            "C": np.random.randint(0, 3, 5),
            "value": np.random.randn(5)
        }))

        predictions = smurff.bpmf(Y,
                                Ytest=Ytest,
                                num_latent=4,
                                verbose=verbose,
                                burnin=50,
                                nsamples=50)

    def test_bpmf_sparse_matrix_sparse_2d_tensor(self):
        np.random.seed(1234)

        # Generate train matrix rows, cols and vals
        train_shape = (5, 5)
        train_rows = np.random.randint(0, 5, 7)
        train_cols = np.random.randint(0, 4, 7)
        train_vals = np.random.randn(7)

        # Generate test matrix rows, cols and vals
        test_shape = (5, 5)
        test_rows = np.random.randint(0, 5, 5)
        test_cols = np.random.randint(0, 4, 5)
        test_vals = np.random.randn(5)

        # Create train and test sparse matrices
        train_sparse_matrix = scipy.sparse.coo_matrix((train_vals, (train_rows, train_cols)), train_shape)
        test_sparse_matrix = scipy.sparse.coo_matrix((test_vals, (test_rows, test_cols)), test_shape)

        # Force NNZ recalculation to remove duplicate coordinates because of random generation
        train_sparse_matrix.count_nonzero()
        test_sparse_matrix.count_nonzero()

        # Create train and test sparse tensors
        train_sparse_tensor = smurff.SparseTensor(pd.DataFrame({
            '0': train_sparse_matrix.row,
            '1': train_sparse_matrix.col,
            'v': train_sparse_matrix.data
        }), train_shape)
        test_sparse_tensor = smurff.SparseTensor(pd.DataFrame({
            '0': test_sparse_matrix.row,
            '1': test_sparse_matrix.col,
            'v': test_sparse_matrix.data
        }), train_shape)

        # Run SMURFF
        sparse_matrix_predictions = smurff.bpmf(train_sparse_matrix,
                                              Ytest=test_sparse_matrix,
                                              num_latent=4,
                                              num_threads=1,
                                              verbose=verbose,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        sparse_tensor_predictions = smurff.bpmf(train_sparse_tensor,
                                              Ytest=test_sparse_tensor,
                                              num_latent=4,
                                              num_threads=1,
                                              verbose=verbose,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        # Transfrom SMURFF results to dictionary of coords and predicted values
        sparse_matrix_predictions_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_matrix_predictions)
        sparse_tensor_predictions_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_tensor_predictions)

        self.assertEqual(len(sparse_matrix_predictions_dict), len(sparse_tensor_predictions_dict))
        self.assertEqual(sparse_tensor_predictions_dict.keys(), sparse_tensor_predictions_dict.keys())
        for coords, matrix_pred_1sample in sparse_matrix_predictions_dict.items():
            tensor_pred_1sample = sparse_tensor_predictions_dict[coords]
            self.assertAlmostEqual(matrix_pred_1sample, tensor_pred_1sample)

    def test_bpmf_dense_matrix_sparse_2d_tensor(self):
        np.random.seed(1234)

        # Generate train dense matrix
        train_shape = (5 ,5)
        train_sparse_matrix = scipy.sparse.random(5, 5, density=1.0)
        train_dense_matrix = train_sparse_matrix.todense()

        # Generate test sparse matrix
        test_shape = (5, 5)
        test_rows = np.random.randint(0, 5, 5)
        test_cols = np.random.randint(0, 4, 5)
        test_vals = np.random.randn(5)
        test_sparse_matrix = scipy.sparse.coo_matrix((test_vals, (test_rows, test_cols)), test_shape)

        # Create train and test sparse tensors
        train_sparse_tensor = smurff.SparseTensor(pd.DataFrame({
            '0': train_sparse_matrix.row,
            '1': train_sparse_matrix.col,
            'v': train_sparse_matrix.data
        }), train_shape)
        test_sparse_tensor = smurff.SparseTensor(pd.DataFrame({
            '0': test_sparse_matrix.row,
            '1': test_sparse_matrix.col,
            'v': test_sparse_matrix.data
        }), train_shape)

        # Run SMURFF
        sparse_matrix_predictions = smurff.bpmf(train_dense_matrix,
                                              Ytest=test_sparse_matrix,
                                              num_latent=4,
                                              num_threads=1,
                                              verbose=verbose,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        sparse_tensor_predictions = smurff.bpmf(train_sparse_tensor,
                                              Ytest=test_sparse_tensor,
                                              num_latent=4,
                                              num_threads=1,
                                              verbose=verbose,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        # Transfrom SMURFF predictions to dictionary of coords and predicted values
        sparse_matrix_predictions_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_matrix_predictions)
        sparse_tensor_predictions_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_tensor_predictions)

        self.assertEqual(len(sparse_matrix_predictions_dict), len(sparse_tensor_predictions_dict))
        self.assertEqual(sparse_tensor_predictions_dict.keys(), sparse_tensor_predictions_dict.keys())
        for coords, matrix_pred_1sample in sparse_matrix_predictions_dict.items():
            tensor_pred_1sample = sparse_tensor_predictions_dict[coords]
            self.assertAlmostEqual(matrix_pred_1sample, tensor_pred_1sample)

    def test_bpmf_tensor2(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(20, 2)
        C = np.random.randn(3, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        predictions = smurff.bpmf(Ytrain,
                                Ytest=Ytest,
                                num_latent=4,
                                verbose=verbose,
                                burnin=20,
                                nsamples=20)

        rmse = smurff.calc_rmse(predictions)

        self.assertTrue(rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % rmse)

    def test_bpmf_tensor3(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(20, 2)
        C = np.random.randn(1, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        predictions = smurff.bpmf(Ytrain,
                                Ytest=Ytest,
                                num_latent=4,
                                verbose=verbose,
                                burnin=20,
                                nsamples=20)

        rmse = smurff.calc_rmse(predictions)

        self.assertTrue(rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % rmse)

        Ytrain_df = Ytrain.data
        Ytest_df = Ytest.data
        Ytrain_sp = scipy.sparse.coo_matrix( (Ytrain_df.value, (Ytrain_df.A, Ytrain_df.B) ) )
        Ytest_sp  = scipy.sparse.coo_matrix( (Ytest_df.value,  (Ytest_df.A, Ytest_df.B) ) )

        results_mat = smurff.bpmf(Ytrain_sp,
                                    Ytest=Ytest_sp,
                                    num_latent=4,
                                    verbose=verbose,
                                    burnin=20,
                                    nsamples=20)

if __name__ == '__main__':
    unittest.main()
