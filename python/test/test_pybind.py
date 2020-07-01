
import smurff
import numpy as np
import scipy.sparse as sp

def test_pybind():
    trainSession = smurff.TrainSession(priors = ["normal", "normal"], verbose = 2 )

    Y = np.array([[1.,2.],[3.,4.]])
    trainSession.setTrain(Y)
    trainSession.setTest(sp.csr_matrix(Y))
    results = trainSession.run()
    # for r in results:
    #     print(r)
