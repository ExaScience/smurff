#!/usr/bin/env python

import smurff

ps = smurff.PredictSession("movielens.hdf5")
sideinfo = smurff.matrix_io.read_matrix("features_1k_random.sdm").tocsr()

print(ps)


print("all:       ", ps.predict_all().shape)
def test(samples):
    print(" == samples ", samples, " ==")
    print("slice:     ", ps.predict( ( slice(10), slice(10) ), samples).shape)
    print("range:     ", ps.predict( ( range(10), range(10) ), samples).shape)
    print("int+range: ", ps.predict( ( 10, range(10) ), samples).shape)
    print("range+int: ", ps.predict( ( range(10), 10 ), samples).shape)

    print("side+int:  ", ps.predict( ( sideinfo, 10 ), samples).shape)
    print("side+int:  ", ps.predict( ( sideinfo[10,:], 10 ), samples ).shape)
    print("side+int:  ", ps.predict( ( sideinfo[10:20,:], 10 ), samples ).shape)

    print("side+range:", ps.predict( ( sideinfo, range(10) ), samples).shape)
    print("side+range:", ps.predict( ( sideinfo[10,:], range(10) ), samples ).shape)
    print("side+range:", ps.predict( ( sideinfo[10:20,:], range(10) ), samples ).shape)

test(None)
test(slice(10))
test(slice(10, 20))
test(slice(10, 20, 2))
