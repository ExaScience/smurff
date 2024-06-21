#include <cstdio>
#include <fstream>
#include <iomanip>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/TrainSession.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/StateFile.h>
#include <SmurffCpp/result.h>

#include "Tests.h"

#define TAG_MATRIX_TESTS "[matrix][random]"
#define TAG_TWO_DIMENTIONAL_TENSOR_TESTS "[tensor2d][random]"
#define TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[tensor3d][random]"

namespace smurff {
namespace test {

// Code for printing test results that can then be copy-pasted into tests as
// expected results
static void printActualResults(int nr, double actualRmseAvg, const std::vector<smurff::ResultItem> &actualResults) {
  static const char *fname = "TestsSmurff_ExpectedResults.h.generated";
  static bool cleanup = true;

  if (cleanup) {
    std::remove(fname);
    cleanup = false;
  }

  std::ofstream os(fname, std::ofstream::app);

  os << "{ " << nr << ",\n"
     << "  { " << std::fixed << std::setprecision(16) << actualRmseAvg << "," << std::endl
     << "      {\n";

  auto sortedResults = actualResults;
  std::sort(sortedResults.begin(), sortedResults.end());

  for (const auto &actualResultItem : actualResults) {
    os << std::setprecision(16);
    os << "         { { " << actualResultItem.coords << " }, " << actualResultItem.val << ", " << std::fixed
       << actualResultItem.pred_1sample << ", " << actualResultItem.pred_avg << ", " << actualResultItem.var << ", "
       << " }," << std::endl;
  }

  os << "      }\n"
     << "  }\n"
     << "},\n";
}

struct ExpectedResult {
  double rmseAvg;
  std::vector<ResultItem> resultItems;
};
std::map<int, ExpectedResult> expectedResults = {
#include "TestsSmurff_ExpectedResults.h"
};


// result comparison
void checkValue(double actualValue, double expectedValue, double epsilon)
{
#ifdef _OPENMP
   double abs_max = std::max(std::abs(actualValue), std::abs(expectedValue));

   if (abs_max > 0)
        CHECK((std::abs(actualValue - expectedValue) / abs_max) < 10.);
   else
        CHECK(std::abs(actualValue - expectedValue) < 10.);
#else
  CHECK(actualValue == Catch::Approx(expectedValue).epsilon(epsilon));
#endif
}

void checkResultItems(const std::vector<ResultItem> &actualResultItems,
                          const std::vector<ResultItem> &expectedResultItems) {
  CHECK(actualResultItems.size() == expectedResultItems.size());

  auto sortedActualResultItems = actualResultItems;
  std::sort(sortedActualResultItems.begin(), sortedActualResultItems.end());
  auto sortedExpectedResultItems = expectedResultItems;
  std::sort(sortedExpectedResultItems.begin(), sortedExpectedResultItems.end());

  for (std::vector<ResultItem>::size_type i = 0; i < sortedActualResultItems.size(); i++)
  {
          const ResultItem &actualResultItem = sortedActualResultItems[i];
          const ResultItem &expectedResultItem = sortedExpectedResultItems[i];

          CHECK(actualResultItem.coords == expectedResultItem.coords);
          CHECK(actualResultItem.val == expectedResultItem.val);

          checkValue(actualResultItem.pred_1sample,expectedResultItem.pred_1sample, single_item_epsilon);
          checkValue(actualResultItem.pred_avg, expectedResultItem.pred_avg, single_item_epsilon);
          checkValue(actualResultItem.var, expectedResultItem.var, single_item_epsilon);
  }
}

struct SmurffTest {
  Config config;

  SmurffTest(const Matrix &train, const SparseMatrix &test, std::vector<PriorTypes> priors)
      : config(genConfig(train, test, priors)) {}

  SmurffTest(const SparseMatrix &train, const SparseMatrix &test, std::vector<PriorTypes> priors)
      : config(genConfig(train, test, priors)) {}

  SmurffTest(const DenseTensor &train, const SparseTensor &test, std::vector<PriorTypes> priors)
      : config(genConfig(train, test, priors)) {}

  SmurffTest(const SparseTensor &train, const SparseTensor &test, std::vector<PriorTypes> priors)
      : config(genConfig(train, test, priors)) {}

  template<class M>
  SmurffTest &addSideInfo(int m, const M &c, bool direct = true) {
    config.addSideInfo(m, makeSideInfoConfig(c, direct));
    return *this;
  }

  SmurffTest &addAuxData(const DataConfig &c) {
    config.addData() = c;
    return *this;
  }


  void runAndCheck(int nr) {
    std::shared_ptr<ISession> trainSession = std::make_shared<TrainSession>(config);
    trainSession->run();

    double actualRmseAvg = trainSession->getRmseAvg();
    const std::vector<ResultItem> &actualResults = trainSession->getResultItems();

    printActualResults(nr, actualRmseAvg, actualResults);
    if(expectedResults.find(nr) == expectedResults.end())
        FAIL("Expected results for nr " << nr << " not found\n");

    double &expectedRmseAvg = expectedResults[nr].rmseAvg;
    auto &expectedResultItems = expectedResults[nr].resultItems;

    checkValue(actualRmseAvg, expectedRmseAvg, rmse_epsilon);
    checkResultItems(actualResults, expectedResultItems);
  }
};

///===========================================================================
TEST_CASE("train_dense_matrix_test_sparse_matrix_normal_normal_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(359);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_normal_normal_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(411);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_normal_normal_dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(467);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_normal_normal_dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(523);
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_spikeandslab_spikeandslab_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(577);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_spikeandslab_spikeandslab_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(629);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_spikeandslab_spikeandslab_dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(685);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_spikeandslab_spikeandslab_dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(741);
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_normalone_normalone_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(795);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_normalone_normalone_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(847);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_normalone_normalone_dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(903);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_normalone_normalone_dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(959);
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_macau_macau_row_side_info_dense_matrix_col_side_info_dense_matrix_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau})
      .addSideInfo(0, rowSideDenseMatrix)
      .addSideInfo(1, colSideDenseMatrix)
      .runAndCheck(1018);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_macau_macau_row_side_info_dense_matrix_col_side_info_dense_matrix_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau})
      .addSideInfo(0, rowSideDenseMatrix)
      .addSideInfo(1, colSideDenseMatrix)
      .runAndCheck(1075);
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_macauone_macauone_row_side_info_sparse_matrix_col_side_info_sparse_matrix_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone})
      .addSideInfo(0, rowSideSparseMatrix)
      .addSideInfo(1, colSideSparseMatrix)
      .runAndCheck(1135);
}

TEST_CASE("train_sparse_matrix_test_sparse_matrix_macauone_macauone_row_side_info_sparse_matrix_col_side_info_sparse_matrix_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone})
      .addSideInfo(0, rowSideSparseMatrix)
      .addSideInfo(1, colSideSparseMatrix)
      .runAndCheck(1193);
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_macau_normal_row_side_info_dense_matrix_none_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal})
      .addSideInfo(0, rowSideDenseMatrix)
      .runAndCheck(1250);
}

//TEST_CASE("train_dense_matrix_test_sparse_matrix_macau_normal_row_side_info_dense_matrix_none_cg",
//          TAG_MATRIX_TESTS) {
//
//  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal})
//      .addSideInfo(0, rowSideDenseMatrix, false)
//      .runAndCheck(1250);
//}

TEST_CASE("train_dense_matrix_test_sparse_matrix_normal_macau_none_col_side_info_dense_matrix_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::macau})
      .addSideInfo(1, colSideDenseMatrix)
      .runAndCheck(1305);
}

//TEST_CASE("train_dense_matrix_test_sparse_matrix_normal_macau_none_col_side_info_dense_matrix_cg",
//          TAG_MATRIX_TESTS) {
//
//  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::macau})
//      .addSideInfo(1, colSideDenseMatrix, false)
//      .runAndCheck(1305);
//}
// test throw - macau prior should have side info

TEST_CASE("train_dense_matrix_test_sparse_matrix_macau_normal_none_none_",
          TAG_MATRIX_TESTS) {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal});
  config.addSideInfo(1, makeSideInfoConfig(rowSideDenseMatrix));

  REQUIRE_THROWS(TrainSession(config).init());
}

// test throw - wrong dimentions of side info

TEST_CASE("train_dense_matrix_test_sparse_matrix_macau_normal_col_side_info_dense_matrix_none_",
          TAG_MATRIX_TESTS) {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal});
  config.addSideInfo(1, makeSideInfoConfig(colSideDenseMatrix));

  REQUIRE_THROWS(TrainSession(config).init());
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_normal_spikeandslab_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::spikeandslab}).runAndCheck(1466);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_spikeandslab_normal_none_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal}).runAndCheck(1518);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_normal_spikeandslab_none_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal})
      .addAuxData(colAuxDense)
      .runAndCheck(1572);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_spikeandslab_normal_dense_matrix_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal})
      .addAuxData(rowAuxDense)
      .runAndCheck(1626);
}

//=================================================================

TEST_CASE("train_dense_matrix_test_sparse_matrix_macau_spikeandslab_row_side_info_dense_matrix_none_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::spikeandslab})
      .addSideInfo(0, rowSideDenseMatrix)
      .runAndCheck(1683);
}

TEST_CASE("train_dense_matrix_test_sparse_matrix_spikeandslab_macau_none_col_side_info_dense_matrix_",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::macau})
      .addSideInfo(1, colSideDenseMatrix)
      .runAndCheck(1738);
}

//=================================================================

TEST_CASE("train_dense_2d_tensor_test_sparse_2d_tensor_normal_normal_none_none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(1792);
}

TEST_CASE("train_sparse_2d_tensor_test_sparse_2d_tensor_normal_normal_none_none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(1844);
}

//=================================================================
TEST_CASE("train_dense_2d_tensor_test_sparse_2d_tensor_spikeandslab_spikeandslab_none_none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(1898);
}

TEST_CASE("train_sparse_2d_tensor_test_sparse_2d_tensor_spikeandslab_spikeandslab_none_none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(1950);
}

//=================================================================

TEST_CASE("train_dense_2d_tensor_test_sparse_2d_tensor_normalone_normalone_none_none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(2004);
}

TEST_CASE("train_sparse_2d_tensor_test_sparse_2d_tensor_normalone_normalone_none_none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(2056);
}

//=================================================================

TEST_CASE("train_dense_3d_tensor_test_sparse_3d_tensor_normal_normal_none_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::normal, PriorTypes::normal, PriorTypes::normal})
      .runAndCheck(2110);
}

TEST_CASE("train_dense_3d_tensor_test_sparse_3d_tensor_spikeandslab_spikeandslab_none_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainDenseTensor3d, testSparseTensor3d,
             {PriorTypes::spikeandslab, PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(2164);
}

TEST_CASE("train_dense_3d_tensor_test_sparse_3d_tensor_macau_normal_row_dense_side_info_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::macau, PriorTypes::normal, PriorTypes::normal})
      .addSideInfo(0, rowSideDenseMatrix3d)
      .runAndCheck(2222);
}

TEST_CASE("train_dense_3d_tensor_test_sparse_3d_tensor_macauone_normal_row_dense_side_info_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[!mayfail]") {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::macauone, PriorTypes::normal, PriorTypes::normal})
      .addSideInfo(0, rowSideDenseMatrix3d)
      .runAndCheck(2280);
}
//=================================================================

TEST_CASE("train_sparse_3d_tensor_test_sparse_3d_tensor_normal_normal_none_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor3d, testSparseTensor3d, {PriorTypes::normal, PriorTypes::normal, PriorTypes::normal})
      .runAndCheck(3110);
}

TEST_CASE("train_sparse_3d_tensor_test_sparse_3d_tensor_spikeandslab_spikeandslab_none_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainSparseTensor3d, testSparseTensor3d,
             {PriorTypes::spikeandslab, PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(3164);
}

TEST_CASE("train_sparse_3d_tensor_test_sparse_3d_tensor_macau_normal_row_dense_side_info_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor3d, testSparseTensor3d, {PriorTypes::macau, PriorTypes::normal, PriorTypes::normal})
      .addSideInfo(0, rowSideDenseMatrix3d)
      .runAndCheck(3222);
}

TEST_CASE("train_sparse_3d_tensor_test_sparse_3d_tensor_macauone_normal_row_dense_side_info_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[!mayfail]") {
  SmurffTest(trainSparseTensor3d, testSparseTensor3d, {PriorTypes::macauone, PriorTypes::normal, PriorTypes::normal})
      .addSideInfo(0, rowSideDenseMatrix3d)
      .runAndCheck(3280);
}
} // namespace test
} // namespace smurff
