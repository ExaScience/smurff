#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#include <boost/version.hpp>

#include <catch2/catch_approx.hpp>

#include <SmurffCpp/result.h>

#include <SmurffCpp/Utils/TruncNorm.h>
#include <SmurffCpp/Utils/InvNormCdf.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

#include <SmurffCpp/Configs/DataConfig.h>

#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.h>

#include <SmurffCpp/Noises/NoiseFactory.h>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/DataMatrices/FullMatrixData.hpp>
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>

#include <SmurffCpp/SideInfo/DenseSideInfo.h>

namespace smurff {


static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE( "mvnormal/rand_gamma", "generaring random gamma variable" ) {
  init_bmrng(1234);
  double g = rand_gamma(100.0, 0.01);
  REQUIRE( g > 0 );
}

TEST_CASE( "latentprior/sample_beta_precision", "sampling beta precision from gamma distribution" ) {
  init_bmrng(1234);
  Matrix beta(2, 3), Lambda_u(2, 2);
  beta << 3.0, -2.00,  0.5,
          1.0,  0.91, -0.2;
  Lambda_u << 0.5, 0.1,
              0.1, 0.3;

  Matrix BBt = beta * beta.transpose();
  auto post = MacauPrior::posterior_beta_precision(BBt, Lambda_u, 0.01, 0.05, beta.cols());
  REQUIRE( post.first  == Catch::Approx(3.005) );
  REQUIRE( post.second == Catch::Approx(0.2631083888) );

  double beta_precision = MacauPrior::sample_beta_precision(BBt, Lambda_u, 0.01, 0.05, beta.cols());
  REQUIRE( beta_precision > 0 );
}

TEST_CASE( "utils/eval_rmse", "Test if prediction variance is correctly calculated")
{
  std::vector<std::uint32_t> rows = {0};
  std::vector<std::uint32_t> cols = {0};
  std::vector<double>        vals = {4.5};

  Model model;

  SparseMatrix S = matrix_utils::sparse_to_eigen(SparseTensor( { 1, 1 }, { rows, cols }, vals));
  std::shared_ptr<Data> data(new ScarceMatrixData(S));
  Result p(S, 3);

  data->setNoiseModel(NoiseFactory::create_noise_model(fixed_ncfg));

  data->init();
  model.init(2, PVec<>({1, 1}), ModelInitTypes::zero, false); //latent dimension has size 2

  auto &t = p.m_predictions.at(0);

  // first iteration
  model.U(0) << 1.0, 0.0;
  model.U(1) << 1.0, 0.0;

  p.update(model, false);

  REQUIRE(t.pred_avg == Catch::Approx(1.0 * 1.0 + 0.0 * 0.0));
  REQUIRE(t.var == Catch::Approx(0.0));
  REQUIRE(p.rmse_1sample == Catch::Approx(std::sqrt(std::pow(4.5 - (1.0 * 1.0 + 0.0 * 0.0), 2) / 1 )));
  REQUIRE(p.rmse_avg ==     Catch::Approx(std::sqrt(std::pow(4.5 - (1.0 * 1.0 + 0.0 * 0.0) / 1, 2) / 1 )));

  //// second iteration
  model.U(0) << 2.0, 0.0;
  model.U(1) << 1.0, 0.0;

  p.update(model, false);

  REQUIRE(t.pred_avg == Catch::Approx(((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0)) / 2));
  REQUIRE(t.var == Catch::Approx(0.5));
  REQUIRE(p.rmse_1sample == Catch::Approx(std::sqrt(std::pow(4.5 - (2.0 * 1.0 + 0.0 * 0.0), 2) / 1 )));
  REQUIRE(p.rmse_avg == Catch::Approx(std::sqrt(std::pow(4.5 - ((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0)) / 2, 2) / 1)));

  //// third iteration

  model.U(0) << 2.0, 0.0;
  model.U(1) << 3.0, 0.0;

  p.update(model, false);

  REQUIRE(t.pred_avg == Catch::Approx(((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0)+ (2.0 * 3.0 + 0.0 * 0.0)) / 3));
  REQUIRE(t.var == Catch::Approx(14.0)); // accumulated variance
  REQUIRE(p.rmse_1sample == Catch::Approx(std::sqrt(std::pow(4.5 - (2.0 * 3.0 + 0.0 * 0.0), 2) / 1 )));
  REQUIRE(p.rmse_avg == Catch::Approx(std::sqrt(std::pow(4.5 - ((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0) + (2.0 * 3.0 + 0.0 * 0.0)) / 3, 2) / 1)));
}

TEST_CASE("utils/auc","AUC ROC") {
  struct TestItem {
      double pred, val;
  };
  std::vector<TestItem> items = {
   { 20.0, 1.0 },
   { 19.0, 0.0 },
   { 18.0, 1.0 },
   { 17.0, 0.0 },
   { 16.0, 1.0 },
   { 15.0, 0.0 },
   { 14.0, 0.0 },
   { 13.0, 1.0 },
   { 12.0, 0.0 },
   { 11.0, 1.0 },
   { 10.0, 0.0 },
   { 9.0,  0.0 },
   { 8.0,  0.0 },
   { 7.0,  0.0 },
   { 6.0,  0.0 },
   { 5.0,  0.0 },
   { 4.0,  0.0 },
   { 3.0,  0.0 },
   { 2.0,  0.0 },
   { 1.0,  0.0 }
  };

  REQUIRE ( calc_auc(items, 0.5) == Catch::Approx(0.84) );
}

TEST_CASE( "ScarceMatrixData/var_total", "Test if variance of Scarce Matrix is correctly calculated") {
  std::vector<std::uint32_t> rows = {0, 1};
  std::vector<std::uint32_t> cols = {0, 0};
  std::vector<double>        vals = {1., 2.};

  const SparseMatrix S = matrix_utils::sparse_to_eigen(SparseTensor( {2, 2}, { rows, cols }, vals));
  std::shared_ptr<Data> data(new ScarceMatrixData(S));

  data->setNoiseModel(NoiseFactory::create_noise_model(fixed_ncfg));

  data->init();
  REQUIRE(data->var_total() == Catch::Approx(0.25));
}

TEST_CASE( "DenseMatrixData/var_total", "Test if variance of Dense Matrix is correctly calculated") {
  Matrix Y(2, 2);
  Y << 1., 2., 3., 4.;

  std::shared_ptr<Data> data(new DenseMatrixData(Y));

  data->setNoiseModel(NoiseFactory::create_noise_model(fixed_ncfg));

  data->init();
  REQUIRE(data->var_total() == Catch::Approx(1.25));
}

using namespace Eigen;
using namespace std;

TEST_CASE("inv_norm_cdf/inv_norm_cdf", "Inverse normal CDF") {
	REQUIRE( inv_norm_cdf(0.0)  == -std::numeric_limits<double>::infinity());
	REQUIRE( inv_norm_cdf(0.5)  == Catch::Approx(0) );
	REQUIRE( inv_norm_cdf(0.9)  == Catch::Approx(1.2815515655446004) );
	REQUIRE( inv_norm_cdf(0.01) == Catch::Approx(-2.3263478740408408) );
}

TEST_CASE("truncnorm/norm_cdf", "Normal CDF") {
	REQUIRE( norm_cdf(0.0)  == Catch::Approx(0.5));
	REQUIRE( norm_cdf(-1.0) == Catch::Approx(0.15865525393145707) );
	REQUIRE( norm_cdf(-3.0) == Catch::Approx(0.0013498980316300933) );
	REQUIRE( norm_cdf(4.0)  == Catch::Approx(0.99996832875816688) );
}

TEST_CASE( "truncnorm/rand_truncnorm", "generaring random truncnorm variable" ) {
  init_bmrng(1234);
  for (int i = 0; i < 10; i++) {
    REQUIRE( rand_truncnorm(2.0) >= 2.0 );
    REQUIRE( rand_truncnorm(3.0) >= 3.0 );
    REQUIRE( rand_truncnorm(5.0) >= 5.0 );
    REQUIRE( rand_truncnorm(50.0) >= 50.0 );
    REQUIRE( rand_truncnorm(30, 2.0, 50.0) >= 50.0 );
  }
}
} // end namespace smurff
