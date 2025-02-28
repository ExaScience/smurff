#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <string>
#include <sstream>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Configs/Config.h>

#include "Tests.h"

namespace smurff {

   namespace test {

TEST_CASE("test assert NA")
{
   auto train = trainDenseMatrix;
   SmurffTest(train, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
}
   } // end namespace test
} // end namespace smurff
