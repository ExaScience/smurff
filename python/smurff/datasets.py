from .prepare import make_train_test

import os
import tempfile
import scipy.io as sio
from hashlib import sha256

try:
    import urllib.request as urllib_request  # for Python 3
except ImportError:
    import urllib2 as urllib_request  # for Python 2

urls = {
        "chembl-IC50-346targets.mm" : 
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm",
            "10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88",
        ),

        "chembl-IC50-compound-feat.mm" : 
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm",
            "f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc",
        ),
}

def load_one(filename):
    (url, expected_sha) =  urls[filename]

    with tempfile.TemporaryDirectory() as tmpdirname:
            output = os.path.join(tmpdirname, filename)
            urllib_request.urlretrieve(url, output)
            actual_sha = sha256(open(output, "rb").read()).hexdigest()
            assert actual_sha == expected_sha
            matrix = sio.mmread(output)

    return matrix

def load_chembl():
    """Downloads a small subset of the ChEMBL dataset.

    Returns
    -------
    ic50_train: sparse matrix
        sparse train matrix

    ic50_test: sparse matrix
        sparse test matrix

    feat: sparse matrix
        sparse row features

    """

    # load bioactivity and features
    ic50 = load_one("chembl-IC50-346targets.mm")
    feat = load_one("chembl-IC50-compound-feat.mm")

    ## creating train and test sets
    ic50_train, ic50_test = make_train_test(ic50, 0.2)

    return (ic50_train, ic50_test, feat)
