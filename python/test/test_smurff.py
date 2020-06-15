import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

class TestSmurff(unittest.TestCase):

    # Python 2.7 @unittest.skip fix
    __name__ = "TestSmurff"

    def test_bpmf(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        predictions = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        self.assertEqual(Ytest.nnz, len(predictions))

    def test_bpmf_numerictest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = 0.3
        X, Xt = smurff.make_train_test(X, Xt)
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['normal', 'normal'],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_macau(self):
        Ydense  = np.random.rand(10, 20)
        r       = np.random.permutation(10*20)[:40] # 40 random samples from 10*20 matrix
        side1   = Ydense[:,1:2]
        side2   = Ydense[1:2,:].transpose()
        Y       = scipy.sparse.coo_matrix(Ydense) # convert to sparse
        Y       = scipy.sparse.coo_matrix( (Y.data[r], (Y.row[r], Y.col[r])), shape=Y.shape )
        Y, Ytest = smurff.make_train_test(Y, 0.5)

        predictions = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['macau', 'macau'],
                                side_info=[side1, side2],
                                direct=True,
                                # side_info_noises=[[('fixed', 1.0, None, None, None)], [('adaptive', None, 0.5, 1.0, None)]],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        #self.assertEqual(Ytest.nnz, len(predictions))

    def test_macau_side_bin(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        F = scipy.sparse.rand(15, 2, 0.5)
        F.data[:] = 1
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['macau', 'normal'],
                      side_info=[F, None],
                      direct=True,
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    def test_macau_dense(self):
        Y  = scipy.sparse.rand(15, 10, 0.2)
        Yt = scipy.sparse.rand(15, 10, 0.1)
        F  = np.random.randn(15, 2)
        smurff.smurff(Y,
                      Ytest=Yt,
                      priors=['macau', 'normal'],
                      side_info=[F, None],
                      direct=True,
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    def test_macau_dense_probit(self):
        A = np.random.randn(25, 2)
        B = np.random.randn(3, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B"])
        df["value"] = (np.array([ np.sum(A[i[0], :] * B[i[1], :]) for i in idx ]) > 0.0).astype(np.float64)
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        threshold = 0.5  # since we sample from mu(0,1)
        
        trainSession = smurff.TrainSession(priors=['macau', 'normal'],
                                num_latent=4,
                                threshold=threshold,
                                burnin=20,
                                nsamples=20,
                                verbose=False)

        trainSession.addTrainAndTest(Ytrain, Ytest, smurff.ProbitNoise(threshold))
        trainSession.addSideInfo(0, A, direct=True)

        predictions = trainSession.run()

        rmse = smurff.calc_rmse(predictions)
        self.assertTrue(rmse > 0.55,
                        msg="Probit factorization (with dense side) gave AUC below 0.55 (%f)." % rmse)

    def test_macau_univariate(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        predictions = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['macauone', 'macauone'],
                                side_info=[side1, side2],
                                direct=True,
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        self.assertEqual(Ytest.nnz, len(predictions))

    def test_too_many_sides(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        with self.assertRaises(AssertionError):
            smurff.smurff(Y,
                          priors=['normal', 'normal', 'normal'],
                          side_info=[None, None, None],
                          verbose = False)

    def test_bpmf_emptytest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        smurff.smurff(X,
                      priors=['normal', 'normal'],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_bpmf_emptytest_probit(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        X.data = X.data > 0.5
        smurff.smurff(X,
                      priors=['normal', 'normal'],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_make_train_test(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xtr, Xte = smurff.make_train_test(X, 0.5)
        self.assertEqual(X.nnz, Xtr.nnz + Xte.nnz)
        diff = np.linalg.norm( (X - Xtr - Xte).todense() )
        self.assertEqual(diff, 0.0)

    def test_make_train_test_df(self):
        idx = list( itertools.product(np.arange(10), np.arange(8), np.arange(3) ))
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.arange(10.0 * 8.0 * 3.0)

        Ytr, Yte = smurff.make_train_test_df(df, 0.4)
        self.assertEqual(Ytr.data.shape[0], df.shape[0] * 0.6)
        self.assertEqual(Yte.data.shape[0], df.shape[0] * 0.4)

        A1 = np.zeros( (10, 8, 3) )
        A2 = np.zeros( (10, 8, 3) )
        A1[df.A, df.B, df.C] = df.value
        A2[Ytr.data.A, Ytr.data.B, Ytr.data.C] = Ytr.data.value
        A2[Yte.data.A, Yte.data.B, Yte.data.C] = Yte.data.value

        self.assertTrue(np.allclose(A1, A2))

    def test_bpmf_tensor(self):
        np.random.seed(1234)
        shape = [5,4,3]

        Y = smurff.SparseTensor(pd.DataFrame({
            "A": np.random.randint(0, 5, 7),
            "B": np.random.randint(0, 4, 7),
            "C": np.random.randint(0, 3, 7),
            "value": np.random.randn(7)
        }),shape)

        Ytest = smurff.SparseTensor(pd.DataFrame({
            "A": np.random.randint(0, 5, 5),
            "B": np.random.randint(0, 4, 5),
            "C": np.random.randint(0, 3, 5),
            "value": np.random.randn(5)
        }),shape)

        predictions = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['normal', 'normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)

    def test_bpmf_tensor2(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(20, 2)
        C = np.random.randn(3, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        predictions = smurff.smurff(Ytrain,
                                Ytest=Ytest,
                                priors=['normal', 'normal', 'normal'],
                                num_latent=4,
                                verbose=False,
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

        predictions = smurff.smurff(Ytrain,
                                Ytest=Ytest,
                                priors=['normal', 'normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=20,
                                nsamples=20)

        rmse = smurff.calc_rmse(predictions)

        self.assertTrue(rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % rmse)

        Ytrain_sp = scipy.sparse.coo_matrix( (Ytrain.data.value, (Ytrain.data.A, Ytrain.data.B) ) )
        Ytest_sp  = scipy.sparse.coo_matrix( (Ytest.data.value,  (Ytest.data.A, Ytest.data.B) ) )

        results_mat = smurff.smurff(Ytrain_sp,
                                    Ytest=Ytest_sp,
                                    priors=['normal', 'normal'],
                                    num_latent=4,
                                    verbose=False,
                                    burnin=20,
                                    nsamples=20)

    def test_macau_tensor(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(3, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        Acoo = scipy.sparse.coo_matrix(A)

        predictions = smurff.smurff(Ytrain = Ytrain, Ytest = Ytest, priors = ['macau', 'normal', 'normal'],
                                side_info =[ Acoo, None, None ],
                                num_latent = 4, verbose = 0, burnin = 20, nsamples = 20)

        rmse = smurff.calc_rmse(predictions)

        self.assertTrue(rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % rmse)

    def test_macau_tensor_univariate(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(3, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        Acoo = scipy.sparse.coo_matrix(A)

        predictions = smurff.MacauSession(Ytrain = Ytrain, Ytest = Ytest, 
                                side_info =[ Acoo, None, None ], univariate = True,
                                num_latent = 4, verbose = 0, burnin = 200, nsamples = 400).run()

        rmse = smurff.calc_rmse(predictions)

        self.assertTrue(rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % rmse)

    def test_macau_tensor_empty(self):
        A = np.random.randn(30, 2)
        B = np.random.randn(4, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])

        Acoo = scipy.sparse.coo_matrix(A)

        predictions = smurff.smurff(smurff.SparseTensor(df),
                           priors=['normal', 'normal', 'normal'],
                           num_latent=2,
                           burnin=5,
                           nsamples=5,
                           verbose=False)

        self.assertFalse(predictions)

    def test_threads(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        for t in range(7): # 1, 2, 4, 8, 16, 32, 64
            smurff.smurff(Y, priors=['normal', 'normal'], num_latent=4, num_threads=2**t, verbose=False, burnin=5, nsamples=5)

if __name__ == '__main__':
    unittest.main()
