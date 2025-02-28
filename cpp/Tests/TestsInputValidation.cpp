#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <string>
#include <sstream>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Configs/Config.h>

#include "Tests.h"

namespace smurff {

   namespace test {

template<typename Matrix, typename Func>
Matrix replaceWithNaN(const Matrix &input, Func f) {
   Matrix output = input;
   f(output);
   return output;
}

template<typename Matrix>
Matrix replaceWithNaN(const Matrix &input);

template<>
Matrix replaceWithNaN(const Matrix &input) {
   return replaceWithNaN(input, [](Matrix &m) { m.setConstant(std::nan("0")); });
}

template<>
SparseMatrix replaceWithNaN(const SparseMatrix &input) {
   return replaceWithNaN(input, [](SparseMatrix &output) {
         // Set all nonzero values to NaN
         for (int k = 0; k < output.outerSize(); ++k)
            for (SparseMatrix::InnerIterator it(output, k); it; ++it)
               it.valueRef() = std::nan("0");
      }
   );
}

template<>
DenseTensor replaceWithNaN(const DenseTensor &input) {
   return replaceWithNaN(input, [](DenseTensor &output) {
      std::fill(output.getValues().begin(), output.getValues().end(), std::nan("0"));
   });
}

template<>
SparseTensor replaceWithNaN(const SparseTensor &input) {
   return replaceWithNaN(input, [](SparseTensor &output) {
      std::fill(output.getValues().begin(), output.getValues().end(), std::nan("0"));
   });
}

template<typename TrainMatrix, typename TestMatrix>
void testNaN(const TrainMatrix &train, const TestMatrix &test) {
   SmurffTest(replaceWithNaN(train), test, {PriorTypes::normal, PriorTypes::normal});
}



TEST_CASE("test assert NA")
{
   CHECK_THROWS(testNaN(trainSparseMatrix, testSparseMatrix));
   CHECK_THROWS(testNaN(trainDenseMatrix, testSparseMatrix));
   CHECK_THROWS(testNaN(trainDenseTensor2d, testSparseTensor2d));
   CHECK_THROWS(testNaN(trainDenseTensor3d, testSparseTensor3d));
   CHECK_THROWS(testNaN(trainSparseTensor2d, testSparseTensor2d));
   CHECK_THROWS(testNaN(trainSparseTensor3d, testSparseTensor3d));
}
   } // end namespace test
} // end namespace smurff
