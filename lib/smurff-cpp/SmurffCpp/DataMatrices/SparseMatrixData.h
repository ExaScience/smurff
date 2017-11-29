#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<double> >
   {
   public:
      SparseMatrixData(Eigen::SparseMatrix<double> Y);

   public:
      double train_rmse(const SubModel& model) const override;
  };
}