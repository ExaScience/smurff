
import logging
import numpy as np
import scipy.sparse as sp

import smurff

# logging.getLogger().setLevel(logging.INFO)

trainSession = smurff.TrainSession(priors = ["normal", "normal"])

Y = np.array([[1.,2.],[3.,4.]])
trainSession.setTrain(Y)
trainSession.setTest(sp.csr_matrix(Y))

results = trainSession.run()
# for r in results:
#     print(r)

