import itertools
import unittest
from numpy.testing._private.utils import assert_equal
from parameterized import parameterized
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import smurff
import scipy.sparse as sp

verbose = 0
nsamples = 25

class TestPredictSession(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestPredictSession"

    def run_train_session(self, nmodes, density):
        shape = range(5, nmodes+5) # 5, 6, 7, ... 
        Y, X = smurff.generate.gen_tensor(shape, 3, density)
        self.Ytrain, self.Ytest = smurff.make_train_test(Y, 0.1)
        priors = ['normal'] * nmodes

        trainSession = smurff.TrainSession(priors = priors, num_latent=4,
                burnin=10, nsamples=nsamples, verbose=verbose,
                save_freq = 1, save_name = smurff.helper.temp_savename())

        trainSession.addTrainAndTest(self.Ytrain, self.Ytest)
        for i, x in enumerate(X):
            trainSession.addSideInfo(i, x)

        trainSession.init()
        while trainSession.step():
            pass

        return trainSession, Y, X

    def run_predict_session(self, nmodes, density):
        train_session, Y, X = self.run_train_session(nmodes, density)
        predict_session = train_session.makePredictSession()
        self.run_predict_some_all_one(train_session, predict_session)
        self.run_predict_predict(predict_session, X)

    def assert_almost_equal_sparse(self, A, B):
        assert_equal(A.shape, B.shape)

        c1,v1 = smurff.find(A)
        c2,v2 = smurff.find(B)
        assert np.array_equal(c1,c2)
        assert np.allclose(v1,v2, atol=0.01) 

    def run_predict_some_all_one(self, train_session, predict_session):
        coords, _ = smurff.find(self.Ytest)
        c0 = [ c[0] for c in coords ]
        p1 = train_session.getTestSamples()
        p2 = predict_session.predict_sparse(self.Ytest)
        p3 = predict_session.predict(c0)
        p4 = predict_session.predict_all()

        # list of scipy.sparse.sp_matrix
        # assert same number of samples
        self.assertEqual(len(p1), nsamples)
        self.assertEqual(len(p2), nsamples)
        self.assertEqual(len(p3), nsamples)
        self.assertEqual(len(p4), nsamples)

        # for all samples
        for s1, s2, s3, s4 in zip(p1, p2, p3, p4):

            # check train_session vs predict_session for all samples
            self.assert_almost_equal_sparse(s1, s2)

            # check predict_session.predict_sparse vs predict_session.predict_one
            coords_s2, values_s2 = smurff.find(s2)
            self.assertEqual([ c[0] for c in coords_s2 ], c0)
            self.assertAlmostEqual(values_s2[0], s3.item(), places=2)

            # check predict_session.predict_sparse vs predict_session.predict_all
            for v, *c in zip(values_s2, *coords_s2):
                self.assertAlmostEqual(v, s4.item(*c), places=2)

    def run_predict_predict(self, predict_session, X):
        """ Test the PredictSession.predict function """

        def run_n_samples(samples, expected_nsamples):
            operand_and_sizes = [ 
                [
                    ( Ellipsis   , x.shape[0] ),
                    ( slice(3)   , 3          ),
                    ( range(3)   , 3          ),
                    ( 3          , 1          ),
                    ( x          , x.shape[0] ),
                    ( x[3]       , 1          ),
                    ( x[1:4]     , 3          ),
                    ( x[0:4:2]   , 2          ),
                ]
            for x in X ]

            MAX = 50
            for c, o in enumerate(itertools.product(*operand_and_sizes)):
                operands, expected_shape = zip(*o) # unzip
                predicted_samples = predict_session.predict(operands, samples)
                self.assertEqual(len(predicted_samples), expected_nsamples, nsamples)
                self.assertEqual(predicted_samples[0].shape, expected_shape)

                if c > MAX: break

        run_n_samples(None, len(predict_session.samples))
        run_n_samples(slice(10), 10)
        run_n_samples(slice(10, 20), 10)
        run_n_samples(slice(10, 20, 2), 5)


    @parameterized.expand(map(lambda x: ("dims%d" % x, x), range(2,5))) # 2, 3, 4
    def test_predict_dense(self, name, nmodes):
        self.run_predict_session(nmodes, 1.0)

    @parameterized.expand(map(lambda x: ("dims%d" % x, x), range(2,5))) # 2, 3, 4
    def test_predict_sparse(self, name, nmodes):
        self.run_predict_session(nmodes, 0.5)

if __name__ == '__main__':
    unittest.main()
