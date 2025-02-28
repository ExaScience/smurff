import unittest
import scipy.sparse
import smurff
import pytest
import numpy as np

verbose = 1

class TestInput(unittest.TestCase):

    # Python 2.7 @unittest.skip fix
    __name__ = "TestInput"

    def assert_na(self, Y):
        with pytest.raises(RuntimeError) as na_error:
            smurff.smurff(Y, priors = ['normal', 'normal'])

        assert("Found na" in str(na_error.value))

    def test_sparse(self):
        Y = scipy.sparse.coo_matrix(( [ np.nan ] , ( [0], [0] )))
        self.assert_na(Y)

    def test_dense(self):
        Y = np.array( [ np.nan ] )
        self.assert_na(Y)

    def test_inf(self):
        Y = np.array( [ np.inf ] )
        self.assert_na(Y)

if __name__ == '__main__':
    unittest.main()
