#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/ConstVMatrixIterator.hpp>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/SaveState.h>
#include <SmurffCpp/Utils/StringUtils.h>

namespace smurff {

Result::Result() {}

Result::Result(const DataConfig &Y, int nsamples)
    : m_dims(Y.getDims()), m_nsamples(nsamples)
{
   if(Y.isDense())
   {
      THROWERROR("test data should be sparse");
   }

   if (Y.isMatrix()) set(Y.getSparseMatrixData(), nsamples);
   else set(Y.getSparseTensorData(), nsamples);
}


//Y - test sparse matrix
Result::Result(const SparseMatrix &Y, int nsamples)
    : m_dims({static_cast<std::uint64_t>(Y.rows()), static_cast<std::uint64_t>(Y.cols())}), m_nsamples(nsamples)
{
    set(Y, nsamples);
}


//Y - test sparse tensor
Result::Result(const SparseTensor &Y, int nsamples)
    : m_dims(Y.getDims()), m_nsamples(nsamples)
{
    set(Y, nsamples);
}

Result::Result(PVec<> lo, PVec<> hi, double value, int nsamples)
    : m_dims((hi - lo).as_uint64_vector()), m_nsamples(nsamples)
{
   for(auto it = PVecIterator(lo, hi); !it.done(); ++it)
   {
      m_predictions.push_back(ResultItem(*it, value, nsamples));
   }
}

void Result::set(const SparseMatrix &Y, int nsamples)
{
   for (int k = 0; k < Y.outerSize(); ++k)
      for (SparseMatrix::InnerIterator it(Y,k); it; ++it)
      {
         PVec<> pos = {it.row(), it.col()};
         m_predictions.push_back(ResultItem(pos, it.value(), nsamples));
      }
}

void Result::set(const SparseTensor &Y, int nsamples)
{
   for(std::uint64_t i = 0; i < Y.getNNZ(); i++)
   {
      const auto p = Y.get(i);
      m_predictions.push_back(ResultItem(p.first, p.second, nsamples));
   }
}

void Result::init()
{
   total_pos = 0;
   if (classify)
   {
      for (const auto &p : m_predictions)
      {
         int is_positive = p.val > threshold;
         total_pos += is_positive;
      }
   }
}


std::vector<SparseMatrix> Result::asVectorOfMatrix() const
{
   std::vector<SparseMatrix> ret;

   for(int i=0; i<m_nsamples; ++i)
   {
      auto pred = toMatrix([i](const ResultItem &p) { return p.pred_all.at(i); });
      ret.push_back(*pred);
   }

   return ret;
}

std::vector<SparseTensor> Result::asVectorOfTensor() const
{
   std::vector<SparseTensor> ret;

   for(int i=0; i<m_nsamples; ++i)
   {
      auto pred = toTensor([i](const ResultItem &p) { return p.pred_all.at(i); });
      ret.push_back(*pred);
   }

   return ret;
}


//--- output model to files

template<typename Accessor>
std::shared_ptr<const SparseMatrix> Result::toMatrix(const Accessor &acc) const
{
   THROWERROR_ASSERT(m_dims.size() == 2);

   auto ret = std::make_shared<SparseMatrix>(m_dims.at(0), m_dims.at(1));
   
   std::vector<Eigen::Triplet<smurff::float_type>> triplets;

   for (const auto &p : m_predictions)
      triplets.push_back({ (int)p.coords.at(0), (int)p.coords.at(1), acc(p) });
   
   ret->setFromTriplets(triplets.begin(), triplets.end());
   return ret;
}

template<typename Accessor>
std::shared_ptr<const SparseTensor> Result::toTensor(const Accessor &acc) const
{
   THROWERROR_ASSERT(m_dims.size() > 2);

   SparseTensor::columns_type columns(m_dims.size());
   std::vector<SparseTensor::value_type> values;
   const std::vector<SparseTensor::dims_type> dims = m_dims;

   for (const auto &p : m_predictions)
   {
      for(unsigned i=0; i<p.coords.size(); i++)
         columns.at(i).push_back(p.coords.at(i));
      values.push_back(acc(p));
   }
   
   return std::make_shared<SparseTensor>(dims, columns, values);
}

void Result::save(SaveState &sf) const
{
   if (isEmpty())
      return;

   if (m_dims.size() == 2)
   {
      auto pred_avg = toMatrix([](const ResultItem &p) { return p.pred_avg; });
      auto pred_var = toMatrix([](const ResultItem &p) { return p.var; });
      auto pred_1sample = toMatrix([](const ResultItem &p) { return p.pred_1sample; });

      sf.putPredAvgVar(*pred_avg, *pred_var, *pred_1sample);
   }

   sf.putPredState(rmse_avg, rmse_1sample, auc_avg, auc_1sample, sample_iter, burnin_iter);
}

void Result::toCsv(std::string filename) const
{
   std::ofstream predFile;
   predFile.open(filename, std::ios::out);
   THROWERROR_ASSERT_MSG(predFile.is_open(), "Error opening file: " + filename);

   for (std::size_t d = 0; d < m_dims.size(); d++)
      predFile << "coord" << d << ",";

   predFile << "y,pred_1samp,pred_avg,var" << std::endl;

   for (std::vector<ResultItem>::const_iterator it = m_predictions.begin(); it != m_predictions.end(); it++)
   {
      it->coords.save(predFile)
          << "," << std::to_string(it->val)
          << "," << std::to_string(it->pred_1sample)
          << "," << std::to_string(it->pred_avg)
          << "," << std::to_string(it->var)
          << std::endl;
   }

   predFile.close();
}

void Result::restore(const SaveState &sf)
{
   sf.getPredState(rmse_avg, rmse_1sample, auc_avg, auc_1sample, sample_iter, burnin_iter);
}

//--- update RMSE and AUC

//model - holds samples (U matrices)
void Result::update(const Model &model, bool burnin)
{
   if (m_predictions.empty())
      return;

   const size_t NNZ = m_predictions.size();

   if (burnin)
   {
      double se_1sample = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample)
      for(size_t k = 0; k < m_predictions.size(); ++k)
      {
         auto &t = m_predictions.operator[](k);
         t.pred_1sample = model.predict(t.coords); //dot product of i'th columns in each U matrix
         se_1sample += std::pow(t.val - t.pred_1sample, 2);
      }

      burnin_iter++;
      rmse_1sample = std::sqrt(se_1sample / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});
      }
   }
   else
   {
      double se_1sample = 0.0;
      double se_avg = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample, se_avg)
      for(size_t k = 0; k < m_predictions.size(); ++k)
      {
         auto &t = m_predictions.operator[](k);
         const double pred = model.predict(t.coords); //dot product of i'th columns in each U matrix
         t.update(pred);

         se_1sample += std::pow(t.val - pred, 2);
         se_avg += std::pow(t.val - t.pred_avg, 2);
      }

      sample_iter++;
      rmse_1sample = std::sqrt(se_1sample / NNZ);
      rmse_avg = std::sqrt(se_avg / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});

         auc_avg = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_avg < b.pred_avg;});
      }
   }
}

std::ostream &Result::info(std::ostream &os, std::string indent) const
{
   if (!m_predictions.empty())
   {
      std::uint64_t dtotal = 1;
      for(size_t d = 0; d < m_dims.size(); d++)
         dtotal *= m_dims[d];

      double test_fill_rate = 100. * m_predictions.size() / dtotal;

      os << indent << "Test data: " << m_predictions.size();

      os << " [";
      for(size_t d = 0; d < m_dims.size(); d++)
      {
         if(d == m_dims.size() - 1)
            os << m_dims[d];
         else
            os << m_dims[d] << " x ";
      }
      os << "]";

      os << " (" << test_fill_rate << "%)" << std::endl;

      if (classify)
      {
         double pos = 100. * (double)total_pos / (double)m_predictions.size();
         os << indent << "Binary classification threshold: " << threshold << std::endl;
         os << indent << "  " << pos << "% positives in test data" << std::endl;
      }
   }
   else
   {
      os << indent << "Test data: -" << std::endl;

      if (classify)
      {
         os << indent << "Binary classification threshold: " << threshold << std::endl;
         os << indent << "  " << "-" << "% positives in test data" << std::endl;
      }
   }

   return os;
}

bool Result::isEmpty() const
{
   return m_predictions.empty();
}
} // end namespace smurff
