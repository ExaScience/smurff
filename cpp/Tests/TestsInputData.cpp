#include <catch2/catch_test_macros.hpp>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

#include "Tests.h"

namespace smurff {
namespace test {

namespace mu = smurff::matrix_utils;

// noise config for train and test
smurff::NoiseConfig fixed_ncfg(NoiseTypes::fixed);
smurff::NoiseConfig unused_ncfg(NoiseTypes::unused);

// dense train data (matrix/tensor 2d/tensor 3d)
smurff::DenseTensor trainDenseTensor2d({3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.});
smurff::DenseTensor trainDenseTensor3d({2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                    13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
smurff::Matrix trainDenseMatrix(mu::dense_to_eigen(trainDenseTensor2d));

// sparse train data (matrix/tensor 2d)
smurff::SparseTensor trainSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                         {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseTensor trainSparseTensor3d({2, 3, 4},
                                        {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseMatrix trainSparseMatrix(mu::sparse_to_eigen(trainSparseTensor2d));

// sparse test data (matrix/tensor 2d/tensor 3d)
smurff::SparseTensor testSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseTensor testSparseTensor3d({2, 3, 4},
                                        {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseMatrix testSparseMatrix(mu::sparse_to_eigen(testSparseTensor2d));

// aux data
static smurff::Matrix rowAuxDenseMatrix = mu::make_dense({3, 1}, {1., 2., 3.});
static smurff::Matrix colAuxDenseMatrix = mu::make_dense({1, 4}, {1., 2., 3., 4.});

smurff::DataConfig rowAuxDense(rowAuxDenseMatrix, fixed_ncfg, {0, 1});
smurff::DataConfig colAuxDense(colAuxDenseMatrix, fixed_ncfg, {1, 0});

// side info
smurff::Matrix       rowSideDenseMatrix   = mu::make_dense({ 3, 1 }, {1., 2., 3.});
smurff::Matrix       colSideDenseMatrix   = mu::make_dense({ 4, 1 }, {1., 2., 3., 4.});
smurff::SparseMatrix rowSideSparseMatrix  = mu::make_sparse({ 3, 1 }, { {0, 1, 2}, {0, 0, 0} }, {1., 2., 3.});
smurff::SparseMatrix colSideSparseMatrix  = mu::make_sparse({ 4, 1 }, { {0, 1, 2, 3}, {0, 0, 0, 0} }, {1., 2., 3., 4.});
smurff::Matrix       rowSideDenseMatrix3d = mu::make_dense({ 2, 3 }, {1., 2., 3., 4., 5., 6.});

} // namespace test
} // namespace smurff
