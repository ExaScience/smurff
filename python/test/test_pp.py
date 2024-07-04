import unittest
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics import mean_squared_error
import math
import smurff
import itertools
import collections

verbose = 0
seed = 1234

# Taken from BMF_PP/postprocess_posterior_samples
def calc_posteriorMeanPrec(predict_session, axis):
    # collect U/V for all samples
    Us = [ s.latents[axis] for s in predict_session.samples ]

    # stack them and compute mean
    Ustacked = np.stack(Us)
    mu = np.mean(Ustacked, axis = 0)

    # Compute Lambdaariance, first unstack in different way
    Uunstacked = np.squeeze(np.split(Ustacked, Ustacked.shape[1], axis = 1))
    Uprec = [ np.linalg.inv(np.cov(u, rowvar = False)) for u in Uunstacked ]

    # restack, shape: (N, K, K)
    Uprecstacked = np.stack(Uprec, axis = 0)

    return mu, Uprecstacked

class TestPP(unittest.TestCase):
    def test_bmf_pp(self):
        np.random.seed(seed)
        Y = scipy.sparse.rand(30, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5, seed=seed)
        trainSession = smurff.BPMFSession(Y, is_scarce = True, Ytest=Ytest,
                num_latent=4, verbose=verbose, num_threads = 1, burnin=20, nsamples=20, save_freq=1,
                seed = seed, save_name=smurff.helper.temp_savename())
        trainSession.run()
        predict_session = trainSession.makePredictSession()

        sess_rmse = float(predict_session.statsYTest["rmse_avg"])
        Ypred, Yvar = predict_session.predictionsYTest

        Yt_i, Yt_j, Yt_v = scipy.sparse.find(Ytest)
        Yp_i, Yp_j, Yp_v = scipy.sparse.find(Ypred)
        assert (Yp_i == Yt_i).all() and (Yp_j == Yt_j).all()

        calc_rmse = math.sqrt(mean_squared_error(Yt_v, Yp_v))

        self.assertAlmostEqual(sess_rmse, calc_rmse, 4)

        for m in range(predict_session.nmodes):
            calc_mu, calc_Lambda = calc_posteriorMeanPrec(predict_session, m)
            sess_mu, sess_Lambda = predict_session.postMuLambda(m)

            np.testing.assert_almost_equal(calc_mu, sess_mu)
            np.testing.assert_almost_equal(calc_Lambda, sess_Lambda)

            # print("calculated mu: ", calc_mu[0:2,0])
            # print("   trainSession mu: ", sess_mu[0:2,0])

            # print("calculated Lambda ", calc_Lambda.shape, ": ", calc_Lambda[0:2,0:2,1] )
            # print("   trainSession Lambda ", sess_Lambda.shape, ": ", sess_Lambda[0:2,0:2,1] )

if __name__ == '__main__':
    unittest.main()
