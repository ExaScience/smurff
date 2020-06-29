#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <limits>

#include <SmurffCpp/Utils/Distribution.h>

static const int N = 10;
static bool cleanup = true;
static const char *fname = "TestsRandom_ExpectedResults.h";

#include "TestsRandom_ExpectedResults.h"

template<typename T> std::string type_name();
template<>           std::string type_name<double>() { return "double"; }
template<>           std::string type_name<std::uint64_t>() { return "std::uint64_t"; }

template <typename T>
void printActualResults(std::string name, const T actualResults[N]) {
  if (cleanup) {
    std::remove(fname);
    cleanup = false;
  }

  std::ofstream os(fname, std::ofstream::app);

  os << std::endl << "static " << type_name<T>() << " expected_" << name << "[N] = { " << std::endl << "  ";

  for (int i = 0; i < N; i++)
    os << std::fixed << std::setprecision(16) << actualResults[i] << ",";

  os << std::endl << "};" << std::endl;
}

template <typename Func, typename T>
void testRandFunc(const Func generator, const std::string &name, const T expected[N])
{
  T rnd[N];

  smurff::init_bmrng(1234);

  for (int i = 0; i < N; i++) rnd[i] = generator();

  printActualResults(name, rnd);

  for (int i = 0; i < N; i++)
    CHECK(rnd[i] == Approx(expected[i]).epsilon(smurff::approx_epsilon<smurff::float_type>()));
}


TEST_CASE("rand", "[random]")
{
  testRandFunc(smurff::rand, "rand", expected_rand);
}

TEST_CASE("rand_unif", "[random]")
{
  testRandFunc(
    []() -> double { return smurff::rand_unif(.0,1.); },
    "rand_unif", expected_rand_unif);
}


TEST_CASE("rand_normal", "[random]")
{
  testRandFunc(smurff::rand_normal, "rand_normal", expected_rand_normal);
}


TEST_CASE("rand_gamma", " [random]")
{
  testRandFunc([]() -> double { return smurff::rand_gamma(35000.,0.00001333); }, "rand_gamma", expected_rand_gamma);
}
