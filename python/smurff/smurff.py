from .trainsession import TrainSession
from .helper import FixedNoise

class SmurffSession(TrainSession):
    def __init__(self, Ytrain, priors, is_scarce = True, Ytest=None, side_info=None, direct=True, *args, **kwargs):
        TrainSession.__init__(self, priors=priors, *args, **kwargs)
        self.addTrainAndTest(Ytrain, Ytest, is_scarce = is_scarce)

        if side_info is not None:
            nmodes = len(Ytrain.shape)
            assert len(side_info) == nmodes, "Too many side info, got %d, expected %d" % (len(side_info), nmodes)
            for mode, si in enumerate(side_info):
                if si is not None:
                    self.addSideInfo(mode, si, direct=direct)


class MacauSession(SmurffSession):
    """A train trainSession specialized for use with the Macau algorithm
   
    Attributes
    ----------
    Ytrain : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
             Train matrix/tensor 
        
         Ytest : :mod:`scipy.sparse` matrix or :class: `SparseTensor`
             Test matrix/tensor. Mainly used for calculating RMSE.

         side_info : list of :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or None
             Side info matrix/tensor for each dimension
             If there is no side info for a certain mode, pass `None`.
             Each side info should have as many rows as you have elemnts in corresponding dimension of `Ytrain`.

         direct : bool
             Use Cholesky instead of CG solver

         univariate : bool
             Use univariate or multivariate sampling.

         \*\*args: 
             Extra arguments are passed to the :class:`TrainSession`
    """

    def __init__(self,  Ytrain, is_scarce = True, Ytest=None, side_info=None, univariate=False, direct=True, *args, **kwargs):
        nmodes = len(Ytrain.shape)
        priors = ['normal'] * nmodes

        if side_info is not None:
            assert len(side_info) == nmodes
            for d in range(nmodes):
                if side_info[d] is not None:
                    priors[d] = 'macau'

        if univariate:
            priors = [p + "one" for p in priors]

        SmurffSession.__init__(self, Ytrain, priors, is_scarce, Ytest, side_info, direct, *args, **kwargs)


class BPMFSession(MacauSession):
    """A train trainSession specialized for use with the BPMF algorithm

    Attributes
    ----------
    Ytrain : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Train matrix/tensor

        Ytest : :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Test matrix/tensor. Mainly used for calculating RMSE.
             
        univariate : bool
            Use univariate or multivariate sampling.

        \*\*args:
            Extra arguments are passed to the :class:`TrainSession`
    """
    def __init__(self, Ytrain, is_scarce = True, Ytest=None, univariate=False, *args, **kwargs):
         MacauSession.__init__(self, Ytrain, is_scarce, Ytest, None, univariate, *args, **kwargs)


class GFASession(SmurffSession):
    def __init__(self, Views, Ytest=None, *args, noise = FixedNoise(), **kwargs):
        Ytrain = Views[0]
        nmodes = len(Ytrain.shape)
        assert nmodes == 2
        priors = ['normal', 'spikeandslab']

        TrainSession.__init__(self, priors=priors, *args, **kwargs)
        self.addTrainAndTest(Ytrain, Ytest, noise = noise)

        for p in range(1, len(Views)):
            self.addData([0, p], Views[p], noise = noise)

# old API -- for compatibility reasons


def smurff(*args, **kwargs):
    return SmurffSession(*args, **kwargs).run()


def bpmf(*args, **kwargs):
    return BPMFSession(*args, **kwargs).run()


def macau(*args, **kwargs):
    return MacauSession(*args, **kwargs).run()


def gfa(*args, **kwargs):
    return GFASession(*args, **kwargs).run()
