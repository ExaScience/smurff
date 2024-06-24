#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/SideInfo/linop.h>
#include <SmurffCpp/Utils/Distribution.h>

namespace smurff {

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);
static SparseMatrix binarySideInfo = matrix_utils::sparse_to_eigen(
    SparseTensor({6,4}, {
         { 0, 3, 3, 2, 5, 4, 1, 2, 4 },
         { 1, 0, 2, 1, 3, 0, 1, 3, 2 } },
         { 1, 1, 1, 1, 1, 1, 1, 1, 1 })
    );
static SparseMatrix binarySideInfoT = binarySideInfo.transpose();


TEST_CASE( "SparseSideInfo/solve_blockcg", "BlockCG solver (1rhs)" )
{
   SparseSideInfo sf(DataConfig(binarySideInfo, false, fixed_ncfg));
   Matrix B(4, 1), X(4, 1), X_true(4, 1);

   B << 0.56,  0.55,  0.3 , -1.78;
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871;
   int niter = linop::solve_blockcg_1block(X, sf, 0.5, B, 1e-6);
   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Catch::Approx(X_true(i,j)) );
     }
   }
   REQUIRE( niter <= 4);
}

TEST_CASE( "SparseSideInfo/solve_blockcg_1_0", "BlockCG solver (3rhs separately)" )
{
   SparseSideInfo sf(DataConfig(binarySideInfo, false, fixed_ncfg));
   Matrix B(3, 4), X(4, 3), X_true(3, 4);

   B << 0.56,  0.55,  0.3 , -1.78,
        0.34,  0.05, -1.48,  1.11,
        0.09,  0.51, -0.63,  1.59;
   B.transposeInPlace();

   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871,
             1.69333333, -0.12709677, -1.94666667,  0.49483871,
             0.66      , -0.04064516, -0.78      ,  0.65225806;
   X_true.transposeInPlace();

   linop::solve_blockcg(X, sf, 0.5, B, 1e-6, 1, 0);

   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Catch::Approx(X_true(i,j)) );
     }
   }
}


TEST_CASE( "Eigen::MatrixFree::1", "Test linop::AtA_mulB - 1" )
{
  SparseSideInfo sf(DataConfig(binarySideInfo, false, fixed_ncfg));

  Matrix B(3, 4), X(4, 3), X_true(3, 4);

  B << 0.56, 0.55, 0.3, -1.78,
      0.34, 0.05, -1.48, 1.11,
      0.09, 0.51, -0.63, 1.59;
   B.transposeInPlace();

  X_true << 0.35555556, 0.40709677, -0.16444444, -0.87483871,
      1.69333333, -0.12709677, -1.94666667, 0.49483871,
      0.66, -0.04064516, -0.78, 0.65225806;
  X_true.transposeInPlace();

  smurff::linop::solve_blockcg_eigen(X, sf, 0.5, B, 1e-6);

  for (int i = 0; i < X.rows(); i++)
  {
    for (int j = 0; j < X.cols(); j++)
    {
      REQUIRE(X(i, j) == Catch::Approx(X_true(i, j)));
    }
  }
}
} // end namespace smurff
