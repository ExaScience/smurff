#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>
#include <algorithm>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff
{
   struct Tensor
   {
   public:
      typedef std::uint64_t dims_type;
      typedef std::uint32_t index_type;
      typedef double value_type;
   protected:
      Tensor(
          const std::vector<dims_type> &dims,
          const std::vector<value_type> &values)
         : m_dims(dims), m_values(values) {}


   public:
      size_t getNModes() const { return m_dims.size(); }
      const std::vector<dims_type> & getDims() const { return m_dims; };
      const dims_type & getNRow() const { return m_dims.at(0); };
      const dims_type & getNCol() const { return m_dims.at(1); };
      dims_type getNNZ() const { return m_values.size(); }

      const std::vector<value_type>& getValues() const { return m_values; }
      std::vector<value_type>& getValues() { return m_values; }

   private:
      std::vector<dims_type> m_dims;
      std::vector<value_type>        m_values;
   };

   struct DenseTensor : public Tensor
   {
      DenseTensor() : Tensor({}, {}) {}

      DenseTensor(
          const std::vector<dims_type> &dims,
          const std::vector<value_type> &values)
         : Tensor(dims, values) {}
   };

   struct SparseTensor : public Tensor
   {
      typedef std::vector<std::vector<index_type>> columns_type;

      SparseTensor() : Tensor({}, {}) {}

      SparseTensor(
          const std::vector<dims_type> &dims,
          const columns_type &columns,
          const std::vector<value_type> &values)
      : Tensor(dims, values), m_columns(columns) {}


      const std::vector<index_type> &getRows() const { return m_columns.at(0); }
      std::vector<index_type> &getRows() { return m_columns.at(0); }

      const std::vector<index_type> &getCols() const { return m_columns.at(1); }
      std::vector<index_type> &getCols() { return m_columns.at(1); }

      const std::vector<index_type> &getColumn(int i) const { return m_columns.at(i); }
      std::vector<index_type> &getColumn(int i) { return m_columns.at(i); }

      std::pair<PVec<>, value_type> get(dims_type) const;
      void set(dims_type, PVec<>, value_type);

   private:
      columns_type m_columns;
   };
}
