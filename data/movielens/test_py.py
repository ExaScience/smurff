#!/usr/bin/env python

import smurff
import pickle

Y = smurff.matrix_io.read_matrix("ratings_1k.sdm")
Ytrain, Ytest = smurff.prepare.make_train_test(Y, 0.2)
sideinfo = smurff.matrix_io.read_matrix("features_1k.sdm")

# result = smurff.macau(Ytrain = Ytrain, Ytest = Ytest, side_info = [sideinfo, None], verbose = 3)
# result = smurff.bpmf(Ytrain = Ytrain, Ytest = Ytest, verbose = 3)

trainSession = smurff.TrainSession(num_latent=8, verbose=2)
trainSession.addTrainAndTest(Ytrain, Ytest)
trainSession.addSideInfo(0, sideinfo, smurff.FixedNoise(10.))

trainSession.run()
