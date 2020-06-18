#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <array>
#include <vector>
#include <numeric>
#include <algorithm>

#include <SmurffCpp/Utils/Error.h>

namespace smurff
{
   
template <size_t MaxSize = 10>
class PVec
{
 private:
   size_t m_size;
   std::array<std::int64_t, MaxSize> m_v;

 public:
   PVec(size_t size = 0)
       : m_size(size)
   {
      if (m_size > MaxSize)
      {
         THROWERROR_SPEC(std::length_error, "Cannot initialize PVec with size greater than MaxSize");
      }

      std::fill(m_v.begin(), m_v.end(), 0);
   }

   PVec(const std::initializer_list<std::int64_t> &l)
   {
      std::fill(m_v.begin(), m_v.end(), 0);

      m_size = std::distance(l.begin(), l.end());

      if (m_size == 0)
      {
         THROWERROR_SPEC(std::length_error, "Cannot initialize PVec with zero length");
      }

      if (m_size > MaxSize)
      {
         THROWERROR_SPEC(std::length_error, "Cannot initialize PVec with size greater than MaxSize");
      }

      std::copy(l.begin(), l.end(), m_v.begin()); // m_v already has correct size
   }

   // Accept any container convertible to std::int64_t
   template <template <typename, typename...> class T, typename... V>
   PVec(const T<V...> &v)
   {
      std::fill(m_v.begin(), m_v.end(), 0);

      m_size = std::distance(v.begin(), v.end());

      if (m_size == 0)
      {
         THROWERROR_SPEC(std::length_error, "Cannot initialize PVec with zero length");
      }

      if (m_size > MaxSize)
      {
         THROWERROR_SPEC(std::length_error, "Initializer size is greater than MaxSize");
      }

      std::copy(v.begin(), v.end(), m_v.begin()); // m_v already has correct size
   }

 public:
   // meta info
   size_t size() const
   {
      return m_size;
   }

   const std::int64_t &operator[](size_t p) const
   {
      return m_v[p];
   }

   const std::int64_t &at(size_t p) const
   {
      if (p >= m_size)
      {
         std::stringstream ss;
         ss << "Cannot access m_v[" << p << "]";
         THROWERROR_SPEC(std::out_of_range, ss.str());
      }
      return m_v[p];
   }

   std::int64_t &operator[](size_t p)
   {
      return m_v[p];
   }

   std::int64_t &at(size_t p)
   {
      if (p >= m_size)
      {
         std::stringstream ss;
         ss << "Cannot access m_v[" << p << "]";
         THROWERROR_SPEC(std::out_of_range, ss.str());
      }
      return m_v[p];
   }

   PVec operator+(const PVec &other) const
   {
      if (m_size != other.m_size)
      {
         THROWERROR_SPEC(std::length_error, "Both PVec intances must have the same size");
      }

      PVec ret(*this);
      std::transform(m_v.begin(), m_v.begin() + m_size, other.m_v.begin(), ret.m_v.begin(), std::plus<std::int64_t>());
      return ret;
   }

   PVec operator-(const PVec &other) const
   {
      if (m_size != other.m_size)
      {
         THROWERROR_SPEC(std::length_error, "Both PVec intances must have the same size");
      }

      PVec ret(*this);
      std::transform(m_v.begin(), m_v.begin() + m_size, other.m_v.begin(), ret.m_v.begin(), std::minus<std::int64_t>());
      return ret;
   }

   bool operator==(const PVec &other) const
   {
      if (m_size != other.m_size)
         return false;

      auto p1 = std::mismatch(m_v.begin(), m_v.begin() + m_size, other.m_v.begin());
      return p1.first == m_v.begin() + m_size || p1.second == other.m_v.begin() + other.m_size;
   }

   bool operator!=(const PVec &other) const
   {
      return !(*this == other);
   }

   bool operator<(const PVec &other) const
   {
      return in(*this, other);
   }

   bool operator>=(const PVec &other) const
   {
      return in(other, *this);
   }

   bool in(const PVec &start, const PVec &end) const
   {
      if (m_size != start.m_size)
      {
         THROWERROR_SPEC(std::length_error, "All PVec intances must have the same size");
      }

      if (m_size != end.m_size)
      {
         THROWERROR_SPEC(std::length_error, "All PVec intances must have the same size");
      }

      for (size_t i = 0; i < m_size; ++i)
      {
         if (at(i) < start.at(i))
            return false;
         if (at(i) >= end.at(i))
            return false;
      }
      return true;
   }

   std::int64_t dot() const
   {
      return std::accumulate(m_v.begin(), m_v.begin() + m_size, 1LL, std::multiplies<std::int64_t>());
   }

   std::ostream &info(std::ostream &os) const
   {
      os << "[ ";
      for (size_t i = 0; i < m_size; ++i)
         os << at(i) << ((i != m_size - 1) ? " x " : "");
      os << " ]";
      return os;
   }

   std::ostream &save(std::ostream &os) const
   {
      for (size_t i = 0; i < m_size; ++i)
         os << at(i) << ((i != m_size - 1) ? "," : "");
      return os;
   }

   std::vector<std::int64_t> as_vector() const
   {
      return std::vector<std::int64_t>(m_v.begin(), m_v.begin() + m_size);
   }
};

template <size_t MaxSize>
std::ostream &operator<<(std::ostream &os, const PVec<MaxSize> &vec)
{
   for (std::uint64_t m = 0; m < vec.size(); m++)
   {
      os << vec[m] << ", ";
   }

   return os;
}

struct PVecIterator
{
public:
   PVecIterator(PVec<> from, PVec<> to)
      : lo(from), hi(to), pos(lo)
   {
      THROWERROR_ASSERT(from.size() == to.size());
      std::cout << " lo = " << lo << std::endl;
      std::cout << " hi = " << hi << std::endl;
   }

   PVecIterator &operator++()
   {
      std::cout << " pos = " << pos << std::endl;
      for (int i = lo.size() - 1; i >= 0; --i)
      {
         pos[i]++;
         if (pos[i] >= hi[i] && i > 0)
            pos[i] = lo[i];
         else
            break;
      }
      return *this;
   }

   bool done() const
   {
      std::cout << pos <<  " < " << hi << "? " << (int)(pos < hi) << std::endl;
      return !(pos < hi);
   }

   const PVec<> &operator*() const
   {
      return pos;
   }

private:
   PVec<> lo, hi, pos;
};


} // end namespace smurff