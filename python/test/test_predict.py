import itertools
import unittest
from parameterized import parameterized
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import smurff
from pprint import pprint

verbose = 0

class TestPredictSession(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestPredictSession"


    def run_train_session(self, nmodes, density):
        shape = range(11, nmodes+11) # 11, 12, 13, ... 
        Y, X = smurff.generate.gen_tensor(shape, 3, density)
        self.Ytrain, self.Ytest = smurff.make_train_test(Y, 0.1)
        priors = ['normal'] * nmodes

        trainSession = smurff.TrainSession(priors = priors, num_latent=4,
                burnin=10, nsamples=25, verbose=verbose,
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

    def run_predict_some_all_one(self, train_session, predict_session):
        p1 = sorted(train_session.getTestPredictions())
        p2 = sorted(predict_session.predict_some(self.Ytest))
        p3 = predict_session.predict_one(p1[0].coords, p1[0].val)
        p4 = predict_session.predict_all()

        self.assertEqual(len(p1), len(p2))

        # check train_session vs predict_session for Ytest
        self.assertEqual(p1[0].coords, p2[0].coords)
        assert_almost_equal(p1[0].val, p2[0].val, decimal = 2)
        assert_almost_equal(p1[0].pred_1sample, p2[0].pred_1sample, decimal = 2)
        assert_almost_equal(p1[0].pred_avg, p2[0].pred_avg, decimal = 2)

        # check predict_session.predict_some vs predict_session.predict_one
        self.assertEqual(p1[0].coords, p3.coords)
        assert_almost_equal(p1[0].val, p3.val, decimal = 2)
        assert_almost_equal(p1[0].pred_1sample, p3.pred_1sample, decimal = 2)
        assert_almost_equal(p1[0].pred_avg, p3.pred_avg, decimal = 2)

        # check predict_session.predict_some vs predict_session.predict_all
        for s in p2:
            ecoords = (Ellipsis,) + s.coords
            for p in zip(s.pred_all, p4[ecoords]):
                self.assertAlmostEqual(*p, places=2)

        p1_rmse_avg = smurff.calc_rmse(p1)
        p2_rmse_avg = smurff.calc_rmse(p2)

        self.assertAlmostEqual(train_session.getRmseAvg(), p2_rmse_avg, places = 2)
        self.assertAlmostEqual(train_session.getRmseAvg(), p1_rmse_avg, places = 2)

    def run_predict_predict(self, predict_session, X):
        """ Test the PredictSession.predict function """

        def run_n_samples(samples, expected_nsamples):
            operand_and_sizes = [ 
                [
                    ( slice(10)  , 10         ),
                    ( range(10)  , 10         ),
                    ( 10         , 1          ),
                    ( x          , x.shape[0] ),
                    ( x[10]      , 1          ),
                    ( x[8:11]    , 3          ),
                    ( x[2:11:2]  , 5          ),
                ]
            for x in X ]

            for o in itertools.product(*operand_and_sizes):
                operands, expected_sizes = zip(*o) # unzip
                expected_shape = (expected_nsamples,) + expected_sizes
                shape = predict_session.predict(operands, samples).shape
                self.assertEqual(shape, expected_shape)

        run_n_samples(None, len(predict_session.samples))
        run_n_samples(slice(10), 10)
        run_n_samples(slice(10, 20), 10)
        run_n_samples(slice(10, 20, 2), 5)


    @parameterized.expand(map(lambda x: ("dims%d" % x, x), range(2,5))) # 2, 3, ..., 6
    def test_predict_dense(self, name, nmodes):
        self.run_predict_session(nmodes, 1.0)

    @parameterized.expand(map(lambda x: ("dims%d" % x, x), range(2,5))) # 2, 3, ..., 6
    def test_predict_sparse(self, name, nmodes):
        self.run_predict_session(nmodes, 0.5)

if __name__ == '__main__':
    unittest.main()
