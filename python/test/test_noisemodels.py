import numpy as np
import scipy.sparse
import smurff
import pytest

verbose = 0
seed = 1234

# 4 different types of side info
def no_side_info(U):
    return None

def sparse_side_info(U):
    return smurff.make_sparse(U, 0.5, seed=seed)

def binary_side_info(U):
    F = np.digitize(U, bins = [.0])
    F = scipy.sparse.coo_matrix(F)
    return F

def dense_side_info(U):
    return U

# 5 different noise configs
def noise_fixed5():
    return smurff.FixedNoise(5.0)

def noise_fixed10():
    return smurff.FixedNoise(10.0)

def noise_adaptive1():
    return smurff.AdaptiveNoise(1.0, 10)

def noise_adaptive10():
    return smurff.AdaptiveNoise(10.0, 100.0)

def noise_probit():
    return smurff.ProbitNoise(.0)

def train_test(density, nmodes, side_info):
    np.random.seed(seed)
    Us = [ np.random.randn(i*4, 1) for i in range(1,nmodes+1) ]
    subscripts = [ [i+1, 0] for i in range(nmodes) ]
    Y = np.einsum(*[j for i in zip(Us,subscripts) for j in i])
    if density < 1.:
        _, Y = smurff.make_train_test(Y, density, seed=seed)
    Ytrain, Ytest = smurff.make_train_test(Y, 0.5, seed=seed)
    return Ytrain, Ytest, side_info(Us[0])

@pytest.mark.parametrize('density', [1.0, 0.5])
@pytest.mark.parametrize('nmodes', [2, 3, 4])
@pytest.mark.parametrize('side_info', [no_side_info, sparse_side_info, binary_side_info, dense_side_info])
@pytest.mark.parametrize('noise_model', [noise_probit, noise_fixed5, noise_fixed10, noise_adaptive10, noise_adaptive1])
def test_noise_model(density, nmodes, side_info, noise_model):
    Ytrain, Ytest, si = train_test(density, nmodes, side_info)
    nm = noise_model()

    priors = ['normal'] * nmodes
    if si is not None:
        priors[0] = 'macau'

    trainSession = smurff.TrainSession(priors = priors, num_latent=8, burnin=20, nsamples=20, threshold=.0, seed=seed, verbose=verbose)

    trainSession.addTrainAndTest(Ytrain, Ytest, nm)
    if not si is None:
        trainSession.addSideInfo(0, si, smurff.SampledNoise(1.), direct=True)

    trainSession.init()
    while trainSession.step():
        pass

    predictions = trainSession.getTestPredictions()
    assert Ytest.nnz == len(predictions)
    if isinstance(nm, smurff.ProbitNoise):
        assert trainSession.getStatus().auc_avg <= 1.
        assert trainSession.getStatus().auc_avg >= 0.
    else:
        assert trainSession.getRmseAvg() < 10.
