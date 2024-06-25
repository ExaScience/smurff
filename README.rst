SMURFF - Scalable Matrix Factorization Framework
================================================

|GitHub Build Status| |Anaconda-Server Badge|

What is Bayesian Matrix Factorization
-------------------------------------

Matrix factorization is a common machine learning technique for
recommender systems, like books for Amazon or movies for Netflix.

.. figure:: https://raw.githubusercontent.com/ExaScience/smurff/master/docs/_static/matrix_factorization.svg?sanitize=true
   :alt: Matrix Factorizaion

The idea of these methods is to approximate the user-movie rating matrix
R as a product of two low-rank matrices U and V such that R ≈ U × V . In
this way U and V are constructed from the known ratings in R, which is
usually very sparsely filled. The recommendations can be made from the
approximation U × V which is dense. If M × N is the dimension of R then
U and V will have dimensions M × K and N × K.

Bayesian probabilistic matrix factorization (BPMF) has been proven to be
more robust to data-overfitting compared to non-Bayesian matrix
factorization.

What is SMURFF
--------------

SMURFF is a highly optimized and parallelized framework for Bayesian
Matrix and Tensors Factorization. SMURFF supports multiple matrix
factorization methods:

* `BPMF <https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf>`__, the basic
  version;
* `Macau <https://arxiv.org/abs/1509.04610>`__, adding support
  for high-dimensional side information to the factorization;
* `GFA <https://arxiv.org/pdf/1411.5799.pdf>`__, doing Group Factor
  Anaysis.

Macau and BPMF can also perform **tensor** factorization.

Examples
--------

Documentation is generated from Jupyter Notebooks. You can find the
notebooks in `docs/notebooks <docs/notebooks>`__ and the resulting
documentation on
`smurff.readthedocs.io <http://smurff.readthedocs.io>`__

Installation
------------

Using `conda <http://anaconda.org>`__:

.. code:: bash

    conda install -c vanderaa smurff

Compile from source code: see `INSTALL.rst <docs/INSTALL.rst>`__

Contributors
------------

-  Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version,
   Tensor factorization)
-  Tom Vander Aa (OpenMP optimized BPMF, Matrix Cofactorization and GFA,
   Code Reorg)
-  Adam Arany (Probit noise model)
-  Tom Haber (Original BPMF code)
-  Andrei Gedich
-  Ilya Pasechnikov
-  Thanh Le Van (sythetic out-of-matrix prediction example)
-  Xiangju Qin (BPMF using posterior propagation)

Citing SMURFF
-------------

If you are using SMURFF in a scientific publication, please cite the following preprint plus the paper describing the corresponding algorithm:

SMURFF: a High-Performance Framework for Matrix Factorization arXiv preprint `arXiv:1904:02514 <https://arxiv.org/abs/1904.02514>`_

When using pure Bayesian Probabilistic Matrix Factorization, please also cite:

 Salakhutdinov R, Mnih A. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. In Proceedings of the 25th international conference on Machine learning (ICML '08), 2008. ACM, New York, NY, USA, 880-887.

When using Bayesian Factorization with Side Information, please also cite:

 Simm J, Arany Á, Zakeri P, Haber T, Wegner JK, Chupakhin V, Ceulemans H, Moreau Y.  Macau: Scalable Bayesian Factorization with High-Dimensional Side Information Using MCMC Proc. of the Machine Learning for  Signal Processing (MLSP), 2017 IEEE 27th International Workshop on MLSP; 2017; Vol. 2017-September; pp. 1 - 6. Tokyo, Japan.

When using Group Factor Analysis, please also cite:

 Klami A, Virtanen S, Leppäaho E, Kaski S., "Group Factor Analysis," in IEEE Transactions on Neural Networks and Learning Systems, vol. 26, no. 9, pp. 2136-2147, Sept. 2015.


Acknowledgements
----------------

Over the course of the last 5 years, this work has been supported by the EU H2020 FET-HPC projects
EPEEC (contract #801051), ExCAPE (contract #671555) and EXA2CT (contract #610741), and the Flemish Exaptation project.

.. |GitHub Build Status| image:: https://github.com/ExaScience/smurff/actions/workflows/build_linux.yml/badge.svg
   :target: https://github.com/ExaScience/smurff

.. |Anaconda-Server Badge| image:: https://anaconda.org/vanderaa/smurff/badges/version.svg
   :target: https://conda.anaconda.org/vanderaa
