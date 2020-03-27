#include "catch.hpp"

#include <Utils/Error.h>
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


TEST_CASE( "SparseSideInfo/solve_blockcg", "BlockCG solver (1rhs)" ) 
{
   SparseSideInfo sf(DataConfig(binarySideInfo, false, fixed_ncfg));
   Matrix B(4, 1), X(4, 1), X_true(4, 1);
 
   B << 0.56,  0.55,  0.3 , -1.78;
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871;
   int niter = linop::solve_blockcg(X, sf, 0.5, B, 1e-6);
   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Approx(X_true(i,j)) );
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
       REQUIRE( X(i,j) == Approx(X_true(i,j)) );
     }
   }
}

TEST_CASE( "linop/solve_blockcg_dense/fail", "BlockCG solver for dense (3rhs separately) [!hide]" ) 
{
   int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
   int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
   Matrix B(3, 4), X(4, 3), sf(6, 4);
 
    sf = Matrix::Zero(6, 4);
    for (int i = 0; i < 9; i++) {
       sf(rows[i], cols[i]) = 1.0;
    }
 
   B << 0.56,  0.55,  0.3 , -1.78,
        0.34,  0.05, -1.48,  1.11,
        0.09,  0.51, -0.63,  1.59;
   B.transposeInPlace();
 
   // this system is unsolvable
   REQUIRE_THROWS(linop::solve_blockcg(X, sf, 0.5, B, 1e-6, true));
}

TEST_CASE( "linop/solve_blockcg_dense/ok", "BlockCG solver for dense (3rhs separately)" ) 
{
   double reg = 0.5;

   Matrix KK(6, 6);
   KK <<  1.7488399 , -1.92816395, -1.39618642, -0.2769755 , -0.52815529, 0.24624319,
        -1.92816395,  3.34435465,  2.07258617,  0.4417173 ,  0.84673143, -0.35075244,
        -1.39618642,  2.07258617,  2.1623261 ,  0.25923918,  0.64428255, -0.2329581,
        -0.2769755 ,  0.4417173 ,  0.25923918,  0.6147927 ,  0.15112057, -0.00692033,
        -0.52815529,  0.84673143,  0.64428255,  0.15112057,  0.80141217, -0.19682322,
         0.24624319, -0.35075244, -0.2329581 , -0.00692033, -0.19682322, 0.56240547;

   Matrix K = KK.llt().matrixU();

   REQUIRE(((K.transpose() * K) - KK).norm() < 1e-3);

   Matrix X_true(3, 6);
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871, -0.16444444, -0.87483871,
             1.69333333, -0.12709677, -1.94666667,  0.49483871, -1.94666667,  0.49483871,
             0.66      , -0.04064516, -0.78      ,  0.65225806, -0.78      ,  0.65225806;
   X_true.transposeInPlace();

   Matrix B = (K.transpose() * K + Matrix::Identity(6,6) * reg) * X_true;
   Matrix X(6, 3);

   //-- Solves the system (K' * K + reg * I) * X = B for X for m right-hand sides
   linop::solve_blockcg(X, K, 0.5, B, 1e-6, false);

   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Approx(X_true(i,j)).epsilon(APPROX_EPSILON) );
     }
   }
}

TEST_CASE( "Eigen::MatrixFree::1", "Test linop::AtA_mulB - 1" )
{
  linop::AtA A(binarySideInfo, 0.5);

  Matrix B(3, 4), X(3, 4), X_true(3, 4);

  B << 0.56, 0.55, 0.3, -1.78,
      0.34, 0.05, -1.48, 1.11,
      0.09, 0.51, -0.63, 1.59;

  X_true << 0.35555556, 0.40709677, -0.16444444, -0.87483871,
      1.69333333, -0.12709677, -1.94666667, 0.49483871,
      0.66, -0.04064516, -0.78, 0.65225806;

  Eigen::ConjugateGradient<linop::AtA, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
  cg.compute(A);
  X = cg.solve(B.transpose()).transpose();

  for (int i = 0; i < X.rows(); i++)
  {
    for (int j = 0; j < X.cols(); j++)
    {
      REQUIRE(X(i, j) == Approx(X_true(i, j)));
    }
  }
}
} // end namespace smurff
