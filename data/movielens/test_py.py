#!/usr/bin/env python

import smurff
import pickle

Y = smurff.matrix_io.read_matrix("ratings_1k_random.sdm")
Ytrain, Ytest = smurff.prepare.make_train_test(Y, 0.2)
sideinfo = smurff.matrix_io.read_matrix("features_1k_random.sdm")


trainSession = smurff.TrainSession(num_latent=8, burnin = 200, nsamples = 200, verbose=1, save_name = "movielens.hdf5", save_freq = 1)
trainSession.addTrainAndTest(Ytrain, Ytest)
trainSession.addSideInfo(0, sideinfo, smurff.FixedNoise(10.))

trainSession.run()

