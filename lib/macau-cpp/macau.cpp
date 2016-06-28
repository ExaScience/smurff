#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <omp.h>

#include "macau.h"
#include "mvnormal.h"
#include "bpmfutils.h"
#include "latentprior.h"

using namespace std; 
using namespace Eigen;

void Macau::addPrior(unique_ptr<ILatentPrior> & prior) {
  priors.push_back( std::move(prior) );
}

void Macau::setPrecision(double p) {
  alpha = p;
}

void Macau::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
}

void Macau::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Y.resize(nrows, ncols);
  sparseFromIJV(Y, rows, cols, values, N);
  Yt = Y.transpose();
  mean_rating = Y.sum() / Y.nonZeros();
}

void Macau::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
}

double Macau::getRmseTest() { return rmse_test; }

void Macau::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  MatrixXd* U = new MatrixXd(num_latent, Y.rows());
  MatrixXd* V = new MatrixXd(num_latent, Y.cols());
  U->setZero();
  V->setZero();
  samples.push_back( std::move(std::unique_ptr<MatrixXd>(U)) );
  samples.push_back( std::move(std::unique_ptr<MatrixXd>(V)) );
}

Macau::~Macau() {
}

inline double sqr(double x) { return x*x; }

void Macau::run() {
  init();
  if (verbose) {
    std::cout << "Sampling" << endl;
  }

  const int num_rows = Y.rows();
  const int num_cols = Y.cols();
  predictions     = VectorXd::Zero( Ytest.nonZeros() );
  predictions_var = VectorXd::Zero( Ytest.nonZeros() );

  auto start = tick();
  for (int i = 0; i < burnin + nsamples; i++) {
    if (verbose && i == burnin) {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();

    // sample latent vectors
    priors[0]->sample_latents(*samples[0], Yt, mean_rating, *samples[1], alpha, num_latent);
    priors[1]->sample_latents(*samples[1], Y,  mean_rating, *samples[0], alpha, num_latent);

    // Sample hyperparams
    priors[0]->update_prior(*samples[0]);
    priors[1]->update_prior(*samples[1]);

    auto eval = eval_rmse(Ytest, (i < burnin) ? 0 : (i - burnin), predictions, predictions_var, *samples[1], *samples[0], mean_rating);

    auto endi = tick();
    auto elapsed = endi - start;
    double samples_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
    double elapsedi = endi - starti;

    if (verbose) {
      printStatus(i, eval.first, eval.second, elapsedi, samples_per_sec);
    }
    rmse_test = eval.second;
  }
}

void Macau::printStatus(int i, double rmse, double rmse_avg, double elapsedi, double samples_per_sec) {
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  printf("Iter %3d: RMSE: %.4f (1samp: %.4f)  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e]  [took %0.1fs]\n", i, rmse_avg, rmse, samples[0]->norm(), samples[1]->norm(), norm0, norm1, elapsedi);
  // if (!std::isnan(norm0)) printf("U.link(%1.2e) U.lambda(%.1f) ", norm0, priors[0]->getLinkLambda());
  // if (!std::isnan(norm1)) printf("V.link(%1.2e) V.lambda(%.1f)",   norm1, priors[1]->getLinkLambda());
}

Eigen::VectorXd Macau::getStds() {
  VectorXd std(Ytest.nonZeros());
  if (nsamples <= 1) {
    std.setConstant(NAN);
    return std;
  }
  const int n = std.size();
  const double inorm = 1.0 / (nsamples - 1);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    std[i] = sqrt(predictions_var[i] * inorm);
  }
  return std;
}

// assumes matrix (not tensor)
Eigen::MatrixXd Macau::getTestData() {
  MatrixXd coords(Ytest.nonZeros(), 3);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < Ytest.outerSize(); ++k) {
    int idx = Ytest.outerIndexPtr()[k];
    for (SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
      coords(idx, 0) = it.row();
      coords(idx, 1) = it.col();
      coords(idx, 2) = it.value();
      idx++;
    }
  }
  return coords;
}
