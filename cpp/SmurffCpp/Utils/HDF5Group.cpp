#include <iostream>

#include <SmurffCpp/Utils/HDF5Group.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>


namespace smurff {

bool HDF5Group::hasDataSet(const std::string& section, const std::string& tag) const
{
   if (!m_group.exist(section)) return false;
   auto section_group = m_group.getGroup(section);
   return (section_group.exist(tag));
}

bool HDF5Group::hasSection(const std::string& section) const
{
   return m_group.exist(section);
}

h5::Group HDF5Group::addGroup(const std::string& name)
{
   if (!m_group.exist(name)) m_group.createGroup(name);
   return getGroup(name);
}

h5::Group HDF5Group::getGroup(const std::string& name) const
{
   return m_group.getGroup(name);
}

void HDF5Group::read(const std::string& section, const std::string& tag, Matrix &X) const
{
   auto dataset = m_group.getGroup(section).getDataSet(tag);
   std::vector<size_t> dims = dataset.getDimensions();
   X.resize(dims[0], dims[1]);
   dataset.read(X.data());

   if (!X.IsRowMajor)
   {
      Matrix colMajor = Eigen::Map<Eigen::Matrix<
            typename Matrix::Scalar,
            Matrix::RowsAtCompileTime,
            Matrix::ColsAtCompileTime,
            Matrix::ColsAtCompileTime==1?Eigen::ColMajor:Eigen::RowMajor,
            Matrix::MaxRowsAtCompileTime,
            Matrix::MaxColsAtCompileTime>>(X.data(), dims[0], dims[1]);

      std::swap(colMajor, X);
   }
}

void HDF5Group::read(const std::string& section, const std::string& tag, Vector &X) const
{
   auto dataset = m_group.getGroup(section).getDataSet(tag);
   std::vector<size_t> dims = dataset.getDimensions();
   THROWERROR_ASSERT(dims[0] == 1);
   X.resize(dims[1]);
   dataset.read(X.data());
}

void HDF5Group::read(const std::string& section, const std::string& tag, SparseMatrix &X) const
{
   auto sparse_group = m_group.getGroup(section).getGroup(tag); 

   std::string format;
   sparse_group.getAttribute("h5sparse_format").read(format);
   THROWERROR_ASSERT(( SparseMatrix::IsRowMajor && format == "csr") || \
                     (!SparseMatrix::IsRowMajor && format == "csc"));
   
   std::vector<Eigen::Index> shape(2);
   sparse_group.getAttribute("h5sparse_shape").read(shape);
   X.resize(shape.at(0), shape.at(1));
   X.makeCompressed();

   auto data = sparse_group.getDataSet("data");
   THROWERROR_ASSERT(data.getDataType() == h5::AtomicType<SparseMatrix::value_type>());
   X.resizeNonZeros(data.getElementCount());
   data.read(X.valuePtr());

   auto indptr = sparse_group.getDataSet("indptr");
   THROWERROR_ASSERT(indptr.getDataType() == h5::AtomicType<SparseMatrix::Index>());
   indptr.read(X.outerIndexPtr());

   auto indices = sparse_group.getDataSet("indices");
   THROWERROR_ASSERT(indices.getDataType() == h5::AtomicType<SparseMatrix::Index>());
   indices.read(X.innerIndexPtr());
}

void HDF5Group::read(const std::string& section, const std::string& tag, DenseTensor &) const
{
   THROWERROR_NOTIMPL();
}

void HDF5Group::read(const std::string& section, const std::string& tag, SparseTensor &) const
{
   THROWERROR_NOTIMPL();
}

void HDF5Group::write(const std::string& section, const std::string& tag, const Vector &V)
{
   write(section, tag, Matrix(V));
}

void HDF5Group::write(const std::string& section, const std::string& tag, const Matrix &M)
{
   if (!m_group.exist(section))
      m_group.createGroup(section);

   h5::Group group = m_group.getGroup(section);
   h5::DataSet dataset = group.createDataSet<Matrix::Scalar>(tag, h5::DataSpace({static_cast<size_t>(M.rows()),  static_cast<size_t>(M.cols())}));

   Eigen::Ref<
        const Eigen::Matrix<
            Matrix::Scalar,
            Matrix::RowsAtCompileTime,
            Matrix::ColsAtCompileTime,
            Matrix::ColsAtCompileTime==1?Eigen::ColMajor:Eigen::RowMajor,
            Matrix::MaxRowsAtCompileTime,
            Matrix::MaxColsAtCompileTime>,
        0,
        Eigen::InnerStride<1>> row_major(M);

    dataset.write_raw(row_major.data());
}

void HDF5Group::write(const std::string& section, const std::string& tag, const SparseMatrix &X)
{
   if (!m_group.exist(section))
      m_group.createGroup(section);

   h5::Group sparse_group = m_group.getGroup(section).createGroup(tag);

   sparse_group.createAttribute<std::string>("h5sparse_format", (SparseMatrix::IsRowMajor ? "csr" : "csc"));
   std::vector<Eigen::Index> shape{X.rows(), X.cols()};
   sparse_group.createAttribute<Eigen::Index>("h5sparse_shape", h5::DataSpace::From(shape)).write(shape);

   auto data = sparse_group.createDataSet<SparseMatrix::value_type>("data", h5::DataSpace(X.nonZeros()));
   data.write(X.valuePtr());

   auto indptr = sparse_group.createDataSet<SparseMatrix::Index>("indptr", h5::DataSpace(X.outerSize() + 1));
   indptr.write(X.outerIndexPtr());

   auto indices = sparse_group.createDataSet<SparseMatrix::Index>("indices", h5::DataSpace(X.nonZeros()));
   indices.write(X.innerIndexPtr());
}

void HDF5Group::write(const std::string& section, const std::string& tag, const DenseTensor &X)
{
   THROWERROR_NOTIMPL();
}

void HDF5Group::write(const std::string& section, const std::string& tag, const SparseTensor &X)
{
   if (!m_group.exist(section))
      m_group.createGroup(section);

   h5::Group sparse_group = m_group.getGroup(section).createGroup(tag);

   sparse_group.createAttribute<std::string>("h5sparse_format", "coo");
   const auto &shape = X.getDims();
   sparse_group.createAttribute<SparseTensor::dims_type>("h5sparse_shape", h5::DataSpace::From(shape)).write(shape);

   auto data = sparse_group.createDataSet<SparseTensor::value_type>("data", h5::DataSpace(X.getNNZ()));
   data.write(X.getValues().data());

   for(size_t i = 0; i < X.getNModes(); ++i)
   {
      auto indices = sparse_group.createDataSet<SparseTensor::index_type>("indices_" + std::to_string(i), h5::DataSpace(X.getNNZ()));
      indices.write(X.getColumn(i).data());
   }
}


} // end namespace smurff
