#!/usr/bin/env python

from scipy import sparse
import numpy as np
import smurff
import smurff.matrix_io as mio
import unittest
import sys
import os
from tempfile import mkdtemp
from os.path import join
import subprocess

global_verbose = False

def train_session(root, train, test, sideinfo = None):
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    print("save prefix = ", root)
    trainSession = smurff.TrainSession(
                                num_latent=4,
                                burnin=800,
                                nsamples=100,
                                verbose = global_verbose,
                                save_freq=1,
                                save_prefix = root,
                                )
    trainSession.addTrainAndTest(train, test, smurff.FixedNoise(1.0))
    if sideinfo is not None:
        trainSession.addSideInfo(0, sideinfo, smurff.FixedNoise(10.), direct=True)

    predictions = trainSession.run()
    rmse = smurff.calc_rmse(predictions)

    #print("RMSE = %.2f%s" % (rmse, "" if sideinfo is None else " (with sideinfo)" ))
    return rmse

def im_prediction(predict_session, test):
    im_predictions = predict_session.predict_sparse(test)
    rmse = smurff.calc_rmse(im_predictions)
    # print("Macau in-matrix prediction RMSE = %.2f" % rmse )
    # print("Predictions:")
    # for p in im_predictions:
    #     print(p)
    # print()
    return rmse

def smurff_py_oom_prediction(predict_session, sideinfo, test):
    sideinfo = sideinfo.tocsr()
    oom_predictions = [
            predict_session.predict_one((sideinfo[i, :], j), v)
            for i,j,v in zip(*sparse.find(test))
            ]
    rmse =  smurff.calc_rmse(oom_predictions)
    #print("Macau ouf-of-matrix prediction RMSE = %.2f" % smurff.calc_rmse(oom_predictions) )
    #print("Predictions:")
    #for p in oom_predictions:
    #    print(p)
    #print()
    return rmse

def calc_rmse(predfile, test):
    predictions = mio.read_matrix(predfile)

    # extract predictions in test matrix
    selected_predictions = [
        smurff.Prediction((i,j), v, pred_avg = predictions[i,j])
        for i,j,v in zip(*sparse.find(test))
    ]

    return smurff.calc_rmse(selected_predictions)

def smurff_cmd_oom_prediction(root, sideinfo_file, test):
    tmpdir = mkdtemp()

    cmd = "smurff --root %s --save-prefix %s --row-features %s --save-freq -1" % (root, tmpdir, sideinfo_file)
    subprocess.call(cmd, shell=True)

    return calc_rmse(join(tmpdir, "predictions-average.ddm"), test)

def tf_cmd_oom_prediction(root_dir, sideinfo_file, test):
    tmpdir = mkdtemp()
    outfile = join(tmpdir, "predictions-average.ddm")

    sideinfo_file = os.path.abspath(sideinfo_file)
    cmd = "source $HOME/miniconda3/bin/activate tf && python tf_predict.py --samples 1 100 --modeldir %s --out %s %s" % (root_dir, outfile, sideinfo_file)
    subprocess.call(cmd, shell=True)

    return calc_rmse(outfile, test)

def af_cmd_oom_prediction(root_dir, sideinfo, test):
    tmpdir = mkdtemp()
    outfile = join(tmpdir, "predictions-average.ddm")
    sideinfo_file = join(tmpdir, "sideinfo.mm")
    mio.write_matrix(sideinfo_file, sideinfo.todense())

    exec_file = os.path.abspath("./af_predict")
    cmd = "%s --modeldir %s/ --out %s --features %s" % (exec_file, root_dir, outfile, sideinfo_file)
    subprocess.call(cmd, shell=True)

    return calc_rmse(outfile, test)

class TestOom(unittest.TestCase):
    def test_macauoom(self):
        train = mio.read_matrix("train.mm").tocsr()
        test = mio.read_matrix("test.mm").tocsr()
        sideinfo = mio.read_matrix("sideinfo.mm").tocsr()

        bpmf_rmse = train_session(mkdtemp(), train, test, )
        rootdir = mkdtemp()
        macau_rmse = train_session(rootdir, train, test, sideinfo, )

        # make out-of-matrix predictions for rows not in train
        num_nonzeros_train = np.diff(train.indptr)
        test_empty = test[num_nonzeros_train == 0]

        rootfile = join(rootdir, "root.ini")
        predict_session = smurff.PredictSession(rootfile)
        rmse_im = im_prediction(predict_session, test_empty)
        rmse_oom_py = smurff_py_oom_prediction(predict_session, sideinfo, test_empty)

        rmse_oom_cmd = smurff_cmd_oom_prediction(rootfile, "sideinfo.mm", test_empty)
        rmse_oom_tf = tf_cmd_oom_prediction(rootdir, "sideinfo.mm", test_empty)
        rmse_oom_af = af_cmd_oom_prediction(rootdir, sideinfo, test_empty)

        print("bpmf full test : %.2f" % bpmf_rmse)
        print("macau full test: %.2f" % macau_rmse)
        print("in-matrix: %.2f" % rmse_im)
        print("out-of-matrix smurff python: %.2f" % rmse_oom_py)
        print("out-of-matrix smurff cmd: %.2f" % rmse_oom_cmd)
        print("out-of-matrix tf (floats): %.2f" % rmse_oom_tf)
        print("out-of-matrix af (floats): %.2f" % rmse_oom_af)

if __name__ == "__main__":
    for arg in sys.argv:
        if (arg == "-v" or arg == "--verbose"):
            global_verbose = True
    unittest.main()
