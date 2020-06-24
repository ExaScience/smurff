#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <highfive/H5File.hpp>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Utils/HDF5Group.h>

namespace h5 = HighFive;

namespace smurff {
   class HDF5Group
   {
   protected:
      h5::Group m_group;
      h5::Group getGroup(const std::string &) const;
      h5::Group addGroup(const std::string &);

   private:
      template <typename T>
      T getInternal(const std::string &section, const std::string& tag, const T &default_value) const
      {
         T value;

         if (this->getGroup(section).hasAttribute(tag))
            this->getGroup(section).getAttribute(tag).read(value);
         else 
            value = default_value;

         return value;
      }

      template <typename T>
      void putInternal(const std::string &section, const std::string& tag, const T &value)
      {
         this->addGroup(section).createAttribute(tag, value);
      }

   public:
      HDF5Group(h5::Group group) : m_group(group) {}

      virtual ~HDF5Group() {}

      bool hasDataSet(const std::string &section, const std::string& tag) const;
      bool hasSection(const std::string &section) const;

      int         get(const std::string &section, const std::string& tag, const int         &default_value) const
      { return getInternal(section, tag, default_value); }

      size_t      get(const std::string &section, const std::string& tag, const size_t      &default_value) const
      { return getInternal(section, tag, default_value); }

      double      get(const std::string &section, const std::string& tag, const double      &default_value) const
      { return getInternal(section, tag, default_value); }

      bool        get(const std::string &section, const std::string& tag, const bool        &default_value) const
      {
          int default_as_int = default_value;
          return getInternal(section, tag, default_as_int);
      }

      std::string get(const std::string &section, const std::string& tag, const std::string &default_value) const
      { return getInternal(section, tag, default_value); }

      void read(const std::string &section, const std::string& tag, Vector &) const;
      void read(const std::string &section, const std::string& tag, Matrix &) const;
      void read(const std::string &section, const std::string& tag, SparseMatrix &) const;
      void read(const std::string &section, const std::string& tag, DenseTensor &) const;
      void read(const std::string &section, const std::string& tag, SparseTensor &) const;

      void put(const std::string &section, const std::string& tag, const int         &value)
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const size_t      &value)
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const double      &value)
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const bool        &value)
      {
          int int_value = value;
          putInternal(section, tag, int_value);
      }

      void put(const std::string &section, const std::string& tag, const std::string &value)
      { putInternal(section, tag, value); }

      void write(const std::string &section, const std::string& tag, const Vector &);
      void write(const std::string &section, const std::string& tag, const Matrix &);
      void write(const std::string &section, const std::string& tag, const SparseMatrix &);
      void write(const std::string &section, const std::string& tag, const DenseTensor &);
      void write(const std::string &section, const std::string& tag, const SparseTensor &);
   };
}
