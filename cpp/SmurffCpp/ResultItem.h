#pragma once

#include <cmath>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

struct ResultItem;

std::ostream &operator<<(std::ostream& os, const ResultItem& r);

struct ResultItem
{
   ResultItem(const PVec<> &c, double v, double p1s, double pa, double var, int n = 0)
   : coords(c), val(v), pred_1sample(p1s), pred_avg(pa), var(var), nsamples(n), keep_samples(0)
   {
   }


   ResultItem(const PVec<> &c, double v = NAN, int n = 0)
   : coords(c), val(v), pred_1sample(NAN), pred_avg(NAN), var(NAN),
     nsamples(0), keep_samples(n)
   {
      pred_all.resize(n);
   }

   PVec<> coords;

   double val;
   double pred_1sample;
   double pred_avg;
   double var;

   int nsamples;
   int keep_samples;

   std::vector<double> pred_all;

   void update(double pred) {
      pred_all.at(nsamples) = pred;

      nsamples++;
      if (nsamples > 1)
      {
        // update pred_1sample, pred_avg and var
        double delta = pred - pred_avg;
        pred_avg = (pred_avg + delta / nsamples);
        var += delta * (pred - pred_avg);
      }
      else
      {
        pred_avg = pred;
        var = 0;
      }
      pred_1sample = pred;
   }

   bool operator<(const ResultItem &other) const
   {
      return coords.as_vector() < other.coords.as_vector();
   }

   std::string to_string() const
   {
      std::stringstream ss;
      ss << *this;
      return ss.str();
   }

};

inline std::ostream &operator<<(std::ostream& os, const ResultItem& r)
{
   os << r.coords << ": "
      << r.val << ","
      << r.pred_1sample << ","
      << r.pred_avg << ","
      << r.var << "[ ";
      for (auto i: r.pred_all) os << i << ", ";
      os << "]";
      return os;
}

} //end namespace