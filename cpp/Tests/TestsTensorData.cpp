#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <string>
#include <sstream>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/DataTensors/SparseMode.h>
#include <SmurffCpp/DataTensors/TensorData.h>

namespace smurff {

TEST_CASE("test sparse view new 1")
{
   std::vector<std::uint64_t> tensorConfigDims = {2, 3};
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns =
      {
         // 1D
         { 0,  1,  0,  1,  0,  1 },
         // 2D
         { 0,  0,  1,  1,  2,  2 },
      };
   std::vector<double> tensorConfigValues =
      {
         0,  3,  1,  4,  2,  5,
      };
   SparseTensor tensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues);

   Matrix actualMatrix0 = matrix_utils::sparse_to_eigen(tensorConfig);

   //std::cout << actualMatrix0 << std::endl;

   TensorData td(tensorConfig);
   /*
   std::cout << "Tensor Data test" << std::endl;

   for(uint64_t mode = 0; mode < td.nmode(); mode++)
   {
      std::shared_ptr<SparseMode> sview = td.Y(mode);

      std::cout << "===============" << std::endl;

      std::cout << "nplanes: " << sview->getNPlanes() << std::endl;
      std::cout << "nnz: " << sview->getNNZ() << std::endl;
      std::cout << "ncoords: " << sview->getNCoords() << std::endl;
      std::cout << "mode: " << sview->getMode() << std::endl;

      for(std::uint64_t n = 0; n < sview->getNPlanes(); n++) // go through each hyper plane in the dimension
      {
         std::cout << "-----" << std::endl;

         for (std::uint64_t j = sview->beginPlane(n); j < sview->endPlane(n); j++) //go through each value in a plane
         {
            for (std::uint64_t m = 0; m < sview->getNCoords(); m++) //go through each coordinate of a value
            {
               std::cout << sview->getIndices()(j, m) << ", "; //print coordinate
            }

            std::cout << sview->getValues()[j] << std::endl; //print value
         }
      }

      std::cout << std::endl;
   }
   */
}

TEST_CASE("test sparse view new 2")
{
   std::vector<std::uint64_t> tensorConfigDims = { 2, 3, 4 };
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns =
      {
         //  1-st xy plane             //2-nd xy plane           //3-rd xy plane            //4-rd xy plane
         // 1D
         { 0,  1,  0,  1,  0,  1,     0,  1,  0,  1,  0,  1,     0,  1,  0,  1,  0,  1,     0,  1,  0,  1,  0,  1 },
         // 2D
         { 0,  0,  1,  1,  2,  2,     0,  0,  1,  1,  2,  2,     0,  0,  1,  1,  2,  2,     0,  0,  1,  1,  2,  2 },
         // 3D
         { 0,  0,  0,  0,  0,  0,     1,  1,  1,  1,  1,  1,     2,  2,  2,  2,  2,  2,     3,  3,  3,  3,  3,  3 },
      };
   std::vector<double> tensorConfigValues =
      {
         0,  3,  1,  4,  2,  5,     6,  9,  7, 10,  8, 11,    12, 15, 13, 16, 14, 17,    18, 21, 19, 22, 20, 23
      };
   SparseTensor tensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues);

   TensorData td(tensorConfig);
   /*
   std::cout << "Tensor Data test" << std::endl;

   // go through each dimension
   for(uint64_t mode = 0; mode < td.nmode(); mode++)
   {
      std::shared_ptr<SparseMode> sview = td.Y(mode);

      std::cout << "===============" << std::endl;

      std::cout << "nplanes: " << sview->getNPlanes() << std::endl;
      std::cout << "nnz: " << sview->getNNZ() << std::endl;
      std::cout << "ncoords: " << sview->getNCoords() << std::endl;
      std::cout << "mode: " << sview->getMode() << std::endl;

      // go through each hyperplane in the dimension
      for(std::uint64_t h = 0; h < sview->getNPlanes(); h++)
      {
         std::cout << "-----" << std::endl;

         // go through each item
         for(std::uint64_t n = 0; n < sview->nItemsOnPlane(h); n++)
         {
            auto item = sview->item(h, n);
            std::cout << item.first << item.second << std::endl; //print item
         }
      }

      std::cout << std::endl;
   }
   */
}

//smurff

/*
TEST_CASE("bpmfutils/eval_rmse_tensor", "Testing eval_rmse_tensor") {
  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       1, 0, 1,
       2, 3, 0;
  Vector v(5);
  v << 0.1, 0.2, 0.3, 0.4, 0.5;

  // mode 0
  SparseMode sm0(C, v, 0, 4);
  int nlatent = 5;
  double gmean = 0.9;

  std::vector< std::unique_ptr<Matrix> > samples;
  Eigen::VectorXi dims(3);
  dims << 4, 5, 2;

  for (int d = 0; d < 3; d++) {
    Matrix* x = new Matrix(nlatent, dims(d));
    rand_normal(*x);
    samples.push_back( std::move(std::unique_ptr<Matrix>(x)) );
  }

  Vector pred(5);
  Vector pred_var(5);
  pred.setZero();
  pred_var.setZero();

  eval_rmse_tensor(sm0, 0, pred, pred_var, samples, gmean);

  for (int i = 0; i < C.rows(); i++) {
    auto v0 = gmean + samples[0]->row(C(i, 0)).
                  cwiseProduct( samples[1]->row(C(i, 1)) ).
                  cwiseProduct( samples[2]->row(C(i, 2)) ).sum();
    REQUIRE(v0 == Catch::Approx(pred(i)));
  }
}

TEST_CASE("latentprior/sample_tensor", "Test whether sampling tensor is correct") {
  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       2, 3, 0,
       1, 0, 1;
  Vector v(5);
  v << 0.15, 0.23, 0.31, 0.47, 0.59;

	Vector mu(3);
	Matrix Lambda(3, 3);
	mu << 0.03, -0.08, 0.12;
	Lambda << 1.2, 0.11, 0.17,
				    0.11, 1.4, 0.08,
						0.17, 0.08, 1.7;

	double mvalue = 0.2;
	double alpha  = 7.5;
  int nlatent = 3;

  std::vector< std::unique_ptr<Matrix> > samples;
	std::vector< std::unique_ptr<SparseMode> > sparseModes;

  Eigen::VectorXi dims(3);
  dims << 4, 5, 2;
  TensorData st(3);
  st.setTrain(C, v, dims);

  for (int d = 0; d < 3; d++) {
    Matrix* x = new Matrix(nlatent, dims(d));
    rand_normal(*x);
    samples.push_back( std::move(std::unique_ptr<Matrix>(x)) );

		SparseMode* sm  = new SparseMode(C, v, d, dims(d));
		sparseModes.push_back( std::move(std::unique_ptr<SparseMode>(sm)) );
  }

	VectorView<Matrix> vv0(samples, 0);
  sample_latent_tensor(samples[0], 0, sparseModes[0], vv0, mvalue, alpha, mu, Lambda);
}

TEST_CASE("macauoneprior/sample_tensor_uni", "Testing sampling tensor univariate") {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat* sf = new SparseFeat(6, 4, 9, rows, cols);
  auto sfptr = std::unique_ptr<SparseFeat>(sf);

  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       2, 3, 0,
       1, 0, 1;
  Vector v(5);
  v << 0.15, 0.23, 0.31, 0.47, 0.59;

	Vector mu(3);
	Matrix Lambda(3, 3);
	mu << 0.03, -0.08, 0.12;
	Lambda << 1.2, 0.11, 0.17,
				    0.11, 1.4, 0.08,
						0.17, 0.08, 1.7;

	double mvalue = 0.2;
	double alpha  = 7.5;
  int nlatent = 3;

  MacauOnePrior<SparseFeat> prior(nlatent, sfptr);

  std::vector< std::unique_ptr<Matrix> > samples;

  Eigen::VectorXi dims(3);
  dims << 6, 5, 2;
  TensorData st(3);
  st.setTrain(C, v, dims);

  for (int d = 0; d < 3; d++) {
    Matrix* x = new Matrix(nlatent, dims(d));
    rand_normal(*x);
    samples.push_back( std::move(std::unique_ptr<Matrix>(x)) );
  }

  prior.sample_latents(alpha, st, samples, 0, nlatent);
}
*/
} // end namespace smurff
