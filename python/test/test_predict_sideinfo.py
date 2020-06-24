import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff

verbose = 0

class TestPredictSession(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestPredictSession"

    def run_train_session(self):
        Ydense  = np.random.normal(size = (10, 20)).reshape((10,20))
        r       = np.random.permutation(10*20)[:40] # 40 random samples from 10*20 matrix
        Y       = scipy.sparse.coo_matrix(Ydense) # convert to sparse
        Y       = scipy.sparse.coo_matrix( (Y.data[r], (Y.row[r], Y.col[r])), shape=Y.shape )

        self.Ytrain, self.Ytest = smurff.make_train_test(Y, 0.5)
        self.side_info   = Ydense


        nmodes = len(self.Ytrain.shape)
        priors = ['normal'] * nmodes

        trainSession = smurff.TrainSession(priors = priors, num_latent=32,
                burnin=10, nsamples=15, verbose=verbose,
                save_freq = 1, save_name=smurff.helper.temp_savename())

        trainSession.addTrainAndTest(self.Ytrain, self.Ytest)
        trainSession.addSideInfo(0, self.side_info, direct=True)
        trainSession.run()
        return trainSession

    def test_simple(self):
        train_session = self.run_train_session()
        predict_session = train_session.makePredictSession()

        p1 = sorted(train_session.getTestPredictions())
        p2 = sorted(predict_session.predict_some(self.Ytest))

        one = p1[0]

        p3 = predict_session.predict_one(one.coords, one.val)
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

        p5 = predict_session.predict([self.side_info[one.coords[0]], one.coords[1]])

        p1_rmse_avg = smurff.calc_rmse(p1)
        p2_rmse_avg = smurff.calc_rmse(p2)

        self.assertAlmostEqual(train_session.getRmseAvg(), p2_rmse_avg, places = 2)
        self.assertAlmostEqual(train_session.getRmseAvg(), p1_rmse_avg, places = 2)
        self.assertAlmostEqual(np.mean(p5), p2_rmse_avg, places = -1)

if __name__ == '__main__':
    unittest.main()
