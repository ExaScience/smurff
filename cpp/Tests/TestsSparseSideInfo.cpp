#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <SmurffCpp/SideInfo/SparseSideInfo.h>
#include <SmurffCpp/SideInfo/linop.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

namespace smurff {

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

SparseMatrix sideInfo = matrix_utils::make_sparse(
    {6, 4},
    {{0, 3, 3, 2, 5, 4, 1, 2, 4}, {1, 0, 2, 1, 3, 0, 1, 3, 2}},
    {0.6, -0.76, 1.48, 1.19, 2.44, 1.95, -0.82, 0.06, 2.54});

TEST_CASE( "SparseSideInfo/At_mul_A", "[At_mul_A] for SparseSideInfo" )
{
    SparseSideInfo si = SparseSideInfo(DataConfig(sideInfo));

    Matrix AA(4, 4);
    si.At_mul_A(AA);
    REQUIRE( AA(0,0) == Catch::Approx(4.3801) );
    REQUIRE( AA(1,1) == Catch::Approx(2.4485) );
    REQUIRE( AA(2,2) == Catch::Approx(8.6420) );
    REQUIRE( AA(3,3) == Catch::Approx(5.9572) );

    REQUIRE( AA(1,0) == 0 );
    REQUIRE( AA(2,0) == Catch::Approx(3.8282) );
    REQUIRE( AA(3,0) == 0 );

    REQUIRE( AA(2,1) == 0 );
    REQUIRE( AA(3,1) == Catch::Approx(0.0714) );

    REQUIRE( AA(3,2) == 0 );
}

TEST_CASE( "SparseSideInfo/A_mul_B", "[A_mul_B] for SparseSideInfo" )
{
    SparseSideInfo si = SparseSideInfo(DataConfig(sideInfo));

    Matrix X(4, 6);
    X << 0., 0.6, 0., 0., 0., -0.82,
        0., 0., 0., 1.19, 0., 0.06,
        -0.76, 0., 1.48, 0., 1.95, 0.,
        2.54, 0., 0., 0., 0., 2.44;

    Matrix Xt = X.transpose();
    Matrix AB = si.A_mul_B(Xt).transpose();

    REQUIRE( AB(0,0) == 0 );
    REQUIRE( AB(1,1) == 0 );
    REQUIRE( AB(2,2) == Catch::Approx(4.953) );
    REQUIRE( AB(3,3) == Catch::Approx(5.9536) );

    REQUIRE( AB(1,0) == Catch::Approx(-0.9044) );
    REQUIRE( AB(2,0) == Catch::Approx(3.8025) );
    REQUIRE( AB(3,0) == 0 );

    REQUIRE( AB(0,1) == Catch::Approx(-0.492) );
    REQUIRE( AB(0,2) == 0 );
    REQUIRE( AB(0,3) == Catch::Approx(-2.0008) );

    REQUIRE( AB(2,1) == Catch::Approx(1.3052) );
    REQUIRE( AB(3,1) == Catch::Approx(1.524) );

    REQUIRE( AB(1,2) == Catch::Approx(1.7612) );
    REQUIRE( AB(1,3) == Catch::Approx(0.1464) );

    REQUIRE( AB(3,2) == 0);

    REQUIRE( AB(2,3) == Catch::Approx(0.0888) );
}

TEST_CASE( "SparseSideInfo/At_mul_Bt", "[At_mul_Bt] for SparseSideInfo" )
{
    SparseSideInfo si = SparseSideInfo(DataConfig(sideInfo));

    Matrix X(4, 6);
    X << 0., 0.6, 0., 0., 0., -0.82,
        0., 0., 0., 1.19, 0., 0.06,
        -0.76, 0., 1.48, 0., 1.95, 0.,
        2.54, 0., 0., 0., 0., 2.44;

    Matrix Xt = X.transpose();

    Vector Y(4);

    si.At_mul_Bt(Y, 0, Xt);

    REQUIRE( Y(0) == 0 );
    REQUIRE( Y(1) == Catch::Approx(-0.9044));
    REQUIRE( Y(2) == Catch::Approx(3.8025) );
    REQUIRE( Y(3) == 0 );
}

TEST_CASE( "SparseSideInfo/add_Acol_mul_bt", "[add_Acol_mul_bt] for SparseSideInfo" )
{
    SparseSideInfo si = SparseSideInfo(DataConfig(sideInfo));

    Matrix Z(4, 6);
    Z << 0., 0.6, 0., 0., 0., -0.82,
        0., 0., 0., 1.19, 0., 0.06,
        -0.76, 0., 1.48, 0., 1.95, 0.,
        2.54, 0., 0., 0., 0., 2.44;

    Matrix Zt = Z.transpose();

    Vector b(4);
    b << 1.4, 0., -0.46, 0.13;

    si.add_Acol_mul_bt(Zt, 2, b);

    Z = Zt.transpose();

    REQUIRE( Z(0,0) == 0 );
    REQUIRE( Z(0,1) == Catch::Approx(0.6) );
    REQUIRE( Z(0,2) == 0 );
    REQUIRE( Z(0,3) == Catch::Approx(2.072) );
    REQUIRE( Z(0,4) == Catch::Approx(3.556) );
    REQUIRE( Z(0,5) == Catch::Approx(-0.82) );

    REQUIRE( Z(1,0) == 0 );
    REQUIRE( Z(1,1) == 0 );
    REQUIRE( Z(1,2) == 0 );
    REQUIRE( Z(1,3) == Catch::Approx(1.19) );
    REQUIRE( Z(1,4) == 0 );
    REQUIRE( Z(1,5) == Catch::Approx(0.06) );

    REQUIRE( Z(2,0) == Catch::Approx(-0.76) );
    REQUIRE( Z(2,1) == 0 );
    REQUIRE( Z(2,2) == Catch::Approx(1.48) );
    REQUIRE( Z(2,3) == Catch::Approx(-0.6808) );
    REQUIRE( Z(2,4) == Catch::Approx(0.7816) );
    REQUIRE( Z(2,5) == 0 );

    REQUIRE( Z(3,0) == Catch::Approx(2.54) );
    REQUIRE( Z(3,1) == 0 );
    REQUIRE( Z(3,2) == 0 );
    REQUIRE( Z(3,3) == Catch::Approx(0.1924) );
    REQUIRE( Z(3,4) == Catch::Approx(0.3302) );
    REQUIRE( Z(3,5) == Catch::Approx(2.44) );
}

TEST_CASE( "SparseSideInfo/col_square_sum", "[col_square_sum] for SparseSideInfo" )
{
    SparseSideInfo si = SparseSideInfo(DataConfig(sideInfo));

    Vector out = si.col_square_sum();

    REQUIRE( out(0) == Catch::Approx(4.3801) );
    REQUIRE( out(1) == Catch::Approx(2.4485) );
    REQUIRE( out(2) == Catch::Approx(8.642) );
    REQUIRE( out(3) == Catch::Approx(5.9572) );
}

TEST_CASE( "SparseSideInfo/compute_uhat", "[compute_uhat] for SparseSideInfo" )
{
    SparseSideInfo si = SparseSideInfo(DataConfig(sideInfo));

    Matrix beta(6,4);
    beta << 1.4, 0., 0.76, 1.34,
            -2.32, 0.12, -1.3, 0.,
            0.45, 0.19, -1.87, 2.34,
            2.12, -1.43, -0.98, -2.71,
            0., 0., 1.10, 2.13,
            0.56, -1.3, 0, 0;

    beta.transposeInPlace();

    Matrix true_uhat(6,6);
    true_uhat << 0, 0, 0.0804, 0.0608, 4.6604, 3.2696,
                0.072, -0.0984, 0.1428, -0.1608, -7.826, 0,
                0.114, -0.1558, 0.3665, -3.1096, -3.8723, 5.7096,
                -0.858, 1.1726, -1.8643, -3.0616, 1.6448, -6.6124,
                0, 0, 0.1278, 1.628, 2.794, 5.1972,
                -0.78, 1.066, -1.547, -0.4256, 1.092, 0;

    true_uhat.transposeInPlace();

    Matrix out(6,6);
    si.compute_uhat(out, beta);

    for (int i = 0; i < true_uhat.rows(); i++) {
        for (int j = 0; j < true_uhat.cols(); j++) {
            REQUIRE( out(i,j) == Catch::Approx(true_uhat(i,j)) );
        }
    }
}

} // end namespace smurff
