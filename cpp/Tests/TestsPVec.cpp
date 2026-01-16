#include <catch2/catch_test_macros.hpp>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

TEST_CASE("PVec<>::PVec(size_t n) | PVec<>::size() | PVec<>::operator[](size_t p)")
{
   PVec<>(0);
   // Fails on some versions of clang macOS
   // REQUIRE_THROWS_AS(PVec<>({}), std::length_error);

   PVec<> p1(1);
   REQUIRE(p1.size() == 1);
   REQUIRE(p1[0] == 0);
   REQUIRE_THROWS_AS(p1.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p1.at(1), std::out_of_range);

   PVec<> p2(2);
   REQUIRE(p2.size() == 2);
   REQUIRE(p2[0] == 0);
   REQUIRE(p2[1] == 0);
   REQUIRE_THROWS_AS(p2.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p2.at(2), std::out_of_range);

   PVec<> p3(3);
   REQUIRE(p3.size() == 3);
   REQUIRE(p3[0] == 0);
   REQUIRE(p3[1] == 0);
   REQUIRE(p3[2] == 0);
   REQUIRE_THROWS_AS(p3.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p3.at(3), std::out_of_range);

   PVec<4> p4(4);
   REQUIRE(p4.size() == 4);
   REQUIRE(p4[0] == 0);
   REQUIRE(p4[1] == 0);
   REQUIRE(p4[2] == 0);
   REQUIRE(p4[3] == 0);
   REQUIRE_THROWS_AS(p4.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p4.at(4), std::out_of_range);
}

TEST_CASE("PVec<>::PVec(const std::initializer_list<int> &l) | PVec<>::size() | PVec<>::operator[](size_t p)")
{
   // Fails on some versions of clang macOS
   // REQUIRE_THROWS_AS(PVec<>({}), std::length_error);

   PVec<> p1({ 1 });
   REQUIRE(p1.size() == 1);
   REQUIRE(p1[0] == 1);
   REQUIRE_THROWS_AS(p1.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p1.at(1), std::out_of_range);

   PVec<> p2({ 1, 2 });
   REQUIRE(p2.size() == 2);
   REQUIRE(p2[0] == 1);
   REQUIRE(p2[1] == 2);
   REQUIRE_THROWS_AS(p2.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p2.at(2), std::out_of_range);

   PVec<> p3({ 1, 2, 3 });
   REQUIRE(p3.size() == 3);
   REQUIRE(p3[0] == 1);
   REQUIRE(p3[1] == 2);
   REQUIRE(p3[2] == 3);
   REQUIRE_THROWS_AS(p3.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p3.at(3), std::out_of_range);

   PVec<4> p4({ 1, 2, 3, 4 });
   REQUIRE(p4.size() == 4);
   REQUIRE(p4[0] == 1);
   REQUIRE(p4[1] == 2);
   REQUIRE(p4[2] == 3);
   REQUIRE(p4[3] == 4);
   REQUIRE_THROWS_AS(p4.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p4.at(4), std::out_of_range);
}

TEST_CASE("PVec<>::PVec(const T<int, V...>& v) | PVec<>::size() | PVec<>::operator[](size_t p)")
{
   std::vector<int> v0;

   std::vector<int> v1 = { 1 };
   PVec<> p1(v1);
   REQUIRE(p1.size() == 1);
   REQUIRE(p1[0] == 1);
   REQUIRE_THROWS_AS(p1.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p1.at(1), std::out_of_range);

   std::vector<int> v2 = { 1, 2 };
   PVec<> p2(v2);
   REQUIRE(p2.size() == 2);
   REQUIRE(p2[0] == 1);
   REQUIRE(p2[1] == 2);
   REQUIRE_THROWS_AS(p2.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p2.at(2), std::out_of_range);

   std::vector<int> v3 = { 1, 2, 3 };
   PVec<> p3(v3);
   REQUIRE(p3.size() == 3);
   REQUIRE(p3[0] == 1);
   REQUIRE(p3[1] == 2);
   REQUIRE(p3[2] == 3);
   REQUIRE_THROWS_AS(p3.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p3.at(3), std::out_of_range);

   std::vector<int> v4 = { 1, 2, 3, 4 };
   PVec<4> p4(v4);
   REQUIRE(p4.size() == 4);
   REQUIRE(p4[0] == 1);
   REQUIRE(p4[1] == 2);
   REQUIRE(p4[2] == 3);
   REQUIRE(p4[3] == 4);
   REQUIRE_THROWS_AS(p4.at(-1), std::out_of_range);
   REQUIRE_THROWS_AS(p4.at(4), std::out_of_range);
}

TEST_CASE("PVec<>::operator==(const PVec& other) | PVec<>::operator!=(const PVec& other)")
{
   PVec<> p0_1({ 1, 2 });
   PVec<> p0_2({ 1, 2 });
   REQUIRE(p0_1 == p0_2);
   REQUIRE_FALSE(p0_1 != p0_2);

   PVec<> p1_1({ 1, 2, 3 });
   PVec<> p1_2({ 1, 2 });
   REQUIRE(p1_1 != p1_2);
   REQUIRE_FALSE(p1_1 == p1_2);

   PVec<9> p2_1({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
   PVec<9> p2_2({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
   REQUIRE(p2_1 == p2_2);
   REQUIRE_FALSE(p2_1 != p2_2);

   PVec<100> p3_1(100);
   PVec<100> p3_2(100);
   REQUIRE(p3_1 == p3_2);
   REQUIRE_FALSE(p3_1 != p3_2);

   PVec<1000> p4_1(100);
   PVec<1000> p4_2(1000);
   REQUIRE(p4_1 != p4_2);
   REQUIRE_FALSE(p4_1 == p4_2);
}

TEST_CASE("PVec<>::operator+(const PVec& other)")
{
   PVec<> p0_1({ 1, 2 });
   PVec<> p0_2({ 3, 4 });
   PVec<> p0_actual = p0_1 + p0_2;
   PVec<> p0_expected({ 4, 6 });
   REQUIRE(p0_actual == p0_expected);

   PVec<> p1_1({ 1, 2 });
   PVec<> p1_2({ 9, -30 });
   PVec<> p1_actual = p1_1 + p1_2;
   PVec<> p1_expected({ 10, -28 });
   REQUIRE(p1_actual == p1_expected);

   PVec<9> p2_1({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
   PVec<9> p2_2({ 1, 1, 1, 1, 0, 0, 0, 0 ,0 });
   PVec<9> p2_actual = p2_1 + p2_2;
   PVec<9> p2_expected({ 2, 3, 4, 5, 5, 6, 7, 8, 9 });
   REQUIRE(p2_actual == p2_expected);

   PVec<6> p3_1({ 1, 2, 3, 4, 5, 6 });
   PVec<6> p3_2(6);
   PVec<6> p3_actual = p3_1 + p3_2;
   PVec<6> p3_expected({ 1, 2, 3, 4, 5, 6 });
   REQUIRE(p3_actual == p3_expected);

   PVec<> p4_1({ 1, 2 });
   PVec<> p4_2({ 1, 2, 3 });
   REQUIRE_THROWS_AS(p4_1 + p4_2, std::length_error);

   PVec<1000> p5_1({ 1, 2, 3, 4, 5, 6 });
   PVec<1000> p5_2(1000);
   REQUIRE_THROWS_AS(p5_1 + p5_2, std::length_error);
}

TEST_CASE("PVec<>::operator-(const PVec& other)")
{
   PVec<> p0_1({ 1, 2 });
   PVec<> p0_2({ 3, 4 });
   PVec<> p0_actual = p0_1 - p0_2;
   PVec<> p0_expected({ -2, -2 });
   REQUIRE(p0_actual == p0_expected);

   PVec<> p1_1({ 1, 2 });
   PVec<> p1_2({ 9, -30 });
   PVec<> p1_actual = p1_1 - p1_2;
   PVec<> p1_expected({ -8, 32 });
   REQUIRE(p1_actual == p1_expected);

   PVec<9> p2_1({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
   PVec<9> p2_2({ 1, 1, 1, 1, 0, 0, 0, 0 ,0 });
   PVec<9> p2_actual = p2_1 - p2_2;
   PVec<9> p2_expected({ 0, 1, 2, 3, 5, 6, 7, 8, 9 });
   REQUIRE(p2_actual == p2_expected);

   PVec<6> p3_1({ 1, 2, 3, 4, 5, 6 });
   PVec<6> p3_2(6);
   PVec<6> p3_actual = p3_1 - p3_2;
   PVec<6> p3_expected({ 1, 2, 3, 4, 5, 6 });
   REQUIRE(p3_actual == p3_expected);

   PVec<> p4_1({ 1, 2 });
   PVec<> p4_2({ 1, 2, 3 });
   REQUIRE_THROWS_AS(p4_1 - p4_2, std::length_error);

   PVec<1000> p5_1({ 1, 2, 3, 4, 5, 6 });
   PVec<1000> p5_2(1000);
   REQUIRE_THROWS_AS(p5_1 - p5_2, std::length_error);
}

TEST_CASE("PVec<>::in(const PVec& start, const PVec& end)")
{
   PVec<> p0({ 4, 5 });
   PVec<> p0_start({ 1, 2, 3 });
   PVec<> p0_end({ 8, 9 });
   REQUIRE_THROWS_AS(p0.in(p0_start, p0_end), std::length_error);

   PVec<> p1({ 4, 5 });
   PVec<> p1_start({ 1, 2 });
   PVec<> p1_end({ 8, 9 });
   REQUIRE(p1.in(p1_start, p1_end));

   PVec<> p2({ 1, 2 });
   PVec<> p2_start({ 4, 5 });
   PVec<> p2_end({ 8, 9 });
   REQUIRE_FALSE(p2.in(p2_start, p2_end));

   PVec<5> p3({ 1, 2, 3, 4, 5 });
   PVec<5> p3_start({ 0, 0, 0, 0, 0 });
   PVec<5> p3_end({ 9, 9, 9, 9, 9 });
   REQUIRE(p3.in(p3_start, p3_end));

   PVec<5> p4({ 1, 2, 9, 4, 5 });
   PVec<5> p4_start({ 0, 0, 0, 0, 0 });
   PVec<5> p4_end({ 9, 9, 9, 9, 9 });
   REQUIRE_FALSE(p4.in(p4_start, p4_end));
}

TEST_CASE("PVec<>::dot()")
{
   PVec<10000> p0(10000);
   REQUIRE(p0.dot() == 0);

   PVec<> p1({ 9 });
   REQUIRE(p1.dot() == 9);

   PVec<> p2({ -9 });
   REQUIRE(p2.dot() == -9);

   PVec<4> p3({ 1, 2, 3, 4 });
   REQUIRE(p3.dot() == 24);

   PVec<4> p4({ 0, 1, 2, 3 });
   REQUIRE(p4.dot() == 0);

   PVec<4> p5({ 1, 2, 3, -4 });
   REQUIRE(p5.dot() == -24);

   PVec<5> p6({ 1, 2, 3, -4, -5 });
   REQUIRE(p6.dot() == 120);
}
} // end namespace smurff
