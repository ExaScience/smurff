import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

verbose = 0

class TestNoiseModels():
    # Python 2.7 @unittest.skip fix
    __name__ = "TestNoiseModels"

    def run_session(self, noise_model):
        Ytrain, Ytest = self.train_test()
        si = self.side_info()

        nmodes = len(Ytrain.shape)
        priors = ['normal'] * nmodes
        if si is not None:
            priors[0] = 'macau'

        trainSession = smurff.TrainSession(priors = priors, num_latent=10, burnin=10, nsamples=15, verbose=verbose)

        if si is None:
            trainSession.addTrainAndTest(Ytrain, Ytest, noise_model)
        elif (isinstance(noise_model, smurff.ProbitNoise) or 
             isinstance(noise_model, smurff.AdaptiveNoise)) and \
             not isinstance(noise_model, smurff.SampledNoise) :
            trainSession.addSideInfo(0, si, direct=True)
            trainSession.addTrainAndTest(Ytrain, Ytest, noise_model)
        else:
            trainSession.addSideInfo(0, si, noise_model, direct=True)
            trainSession.addTrainAndTest(Ytrain, Ytest)

        trainSession.init()
        while trainSession.step():
            pass

        predictions = trainSession.getTestPredictions()
        self.assertEqual(Ytest.nnz, len(predictions))
        self.assertLess(trainSession.getRmseAvg(), 10.)
        return predictions

    # 5 different noise configs

    def test_fixed5(self):
        self.run_session(smurff.FixedNoise(5.0))

    def test_fixed10(self):
        self.run_session(smurff.FixedNoise(10.0))

    def test_adaptive1(self):
        self.run_session(smurff.AdaptiveNoise(1.0, 10))

    def test_adaptive10(self):
        self.run_session(smurff.AdaptiveNoise(10.0, 100.0))

    def test_probit(self):
        self.run_session(smurff.ProbitNoise(0.0))
#
# 2 different types of train&test
#

class TestNoiseModelsMatrix():
   def train_test(self):
       Y = scipy.sparse.rand(15, 10, 0.2)
       Y, Ytest = smurff.make_train_test(Y, 0.5)
       return Y, Ytest

class TestNoiseModelsTensor():
    def train_test(self):
        np.random.seed(1234)
        train_df = pd.DataFrame({
            "A": np.random.randint(0, 15, 7),
            "B": np.random.randint(0, 4,  7),
            "C": np.random.randint(0, 3,  7),
            "value": np.random.randn(7)
        })
        test_df = pd.DataFrame({
            "A": np.random.randint(0, 15, 5),
            "B": np.random.randint(0, 4,  5),
            "C": np.random.randint(0, 3,  5),
            "value": np.random.randn(5)
        })

        shape = [15, 4, 3]
        Ytrain = smurff.SparseTensor(train_df, shape = shape)
        Ytest  = smurff.SparseTensor(test_df, shape = shape)

        return Ytrain, Ytest


#
# 4 different types of side info
#
class TestNoiseModelsBPMF():
    def side_info(self):
        return None

class TestNoiseModelsMacauSparse():
    def side_info(self):
        return scipy.sparse.rand(15, 2, 0.5)

class TestNoiseModelsMacauSparseBin():
    def side_info(self):
        F = scipy.sparse.rand(15, 2, 0.5)
        F.data[:] = 1
        return F
 
class TestNoiseModelsMacauDense():
    def side_info(self):
        return np.random.randn(15, 2)

#
# Now make all combinations
#
class TestNoiseModelsMatrixBPMF(unittest.TestCase, TestNoiseModels, TestNoiseModelsMatrix, TestNoiseModelsBPMF):
    pass

class TestNoiseModelsMatrixMacauSparse(unittest.TestCase, TestNoiseModels, TestNoiseModelsMatrix, TestNoiseModelsMacauSparse):
    pass

class TestNoiseModelsMatrixMacauSparseBin(unittest.TestCase, TestNoiseModels, TestNoiseModelsMatrix, TestNoiseModelsMacauSparseBin):
    pass

class TestNoiseModelsMatrixMacauDense(unittest.TestCase, TestNoiseModels, TestNoiseModelsMatrix, TestNoiseModelsMacauDense):
    pass

class TestNoiseModelsTensorBPMF(unittest.TestCase, TestNoiseModels, TestNoiseModelsTensor, TestNoiseModelsBPMF):
    pass

class TestNoiseModelsTensorMacauSparse(unittest.TestCase, TestNoiseModels, TestNoiseModelsTensor, TestNoiseModelsMacauSparse):
    pass

class TestNoiseModelsTensorMacauSparseBin(unittest.TestCase, TestNoiseModels, TestNoiseModelsTensor, TestNoiseModelsMacauSparseBin):
    pass

class TestNoiseModelsTensorMacauDense(unittest.TestCase, TestNoiseModels, TestNoiseModelsTensor, TestNoiseModelsMacauDense):
    pass
  
if __name__ == '__main__':
    unittest.main()
