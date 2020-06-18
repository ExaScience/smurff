import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff

verbose = 0

class TestPredictSession(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestPredictSession"

    def run_train_session(self, nmodes = 2):
        shape = range(2, nmodes+2) # 2, 3, 4, ... 
        Y = np.random.rand(*shape)
        self.Ytrain, self.Ytest = smurff.make_train_test(Y, 0.5)
        priors = ['normal'] * nmodes

        session = smurff.TrainSession(priors = priors, num_latent=4,
                burnin=10, nsamples=15, verbose=verbose,
                save_freq = 1)

        session.addTrainAndTest(self.Ytrain, self.Ytest)

        session.init()
        while session.step():
            pass

        return session

    def run_predict_session(self, nmodes):
        train_session = self.run_train_session(nmodes)
        predict_session = train_session.makePredictSession()

        p1 = sorted(train_session.getTestPredictions())
        p2 = sorted(predict_session.predict_some(self.Ytest))
        p3 = predict_session.predict_one(p1[0].coords, p1[0].val)
        p4 = predict_session.predict_all()

        self.assertEqual(len(p1), len(p2))

        # check train_session vs predict_session for Ytest
        self.assertEqual(p1[0].coords, p2[0].coords)
        self.assertAlmostEqual(p1[0].val, p2[0].val, places = 2)
        self.assertAlmostEqual(p1[0].pred_1sample, p2[0].pred_1sample, places = 2)
        self.assertAlmostEqual(p1[0].pred_avg, p2[0].pred_avg, places = 2)

        # check predict_session.predict_some vs predict_session.predict_one
        self.assertEqual(p1[0].coords, p3.coords)
        self.assertAlmostEqual(p1[0].val, p3.val, places = 2)
        self.assertAlmostEqual(p1[0].pred_1sample, p3.pred_1sample, places = 2)
        self.assertAlmostEqual(p1[0].pred_avg, p3.pred_avg, places = 2)

        # check predict_session.predict_some vs predict_session.predict_all
        for s in p2:
            ecoords = (Ellipsis,) + s.coords
            for p in zip(s.pred_all, p4[ecoords]):
                self.assertAlmostEqual(*p, places=2)

        p1_rmse_avg = smurff.calc_rmse(p1)
        p2_rmse_avg = smurff.calc_rmse(p2)

        self.assertAlmostEqual(train_session.getRmseAvg(), p2_rmse_avg, places = 2)
        self.assertAlmostEqual(train_session.getRmseAvg(), p1_rmse_avg, places = 2)

    def test_predict(self):
        for nmodes in range(2,7): # 2, 3, ..., 6
            self.run_predict_session(nmodes)

if __name__ == '__main__':
    unittest.main()
