#pragma once

#include "MatrixDataTempl.hpp"

#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <SmurffCpp/Utils/ThreadVector.hpp>

namespace smurff
{
   template<class YType>
   class FullMatrixData : public MatrixDataTempl<YType>
   {
   protected:
      Matrix VV[2]; // sum of v * vT, where v is column of V

   public:
      FullMatrixData(YType Y) 
         : MatrixDataTempl<YType>(Y)
      {
         this->name = "MatrixData [fully known]";
      }

   public:
      //purpose of update_pnm is to cache VV matrix
      void update_pnm(const SubModel& model, uint32_t mode) override
      {
         auto Vf = *model.CVbegin(mode);
         const int nl = model.nlatent();
         thread_vector<Matrix> VVs(Matrix::Zero(nl, nl));

         //for each column v of Vf - calculate v * vT and add to VVs
         #pragma omp parallel for schedule(guided) shared(VVs)
         for(int n = 0; n < Vf.rows(); n++) 
         {
            auto v = Vf.row(n);
            VVs.local() += v.transpose() * v; // VVs = Vvs + vT * v
         }

         VV[mode] = VVs.combine(); //accumulate sum
      }

      std::uint64_t nna() const override
      {
         return 0;
      }
   };
}
