#include "DataConfig.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Utils/HDF5Group.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>

namespace smurff {

static const std::string POS_TAG = "pos";
static const std::string DATA_TAG = "data";
static const std::string DENSE_TAG = "dense";
static const std::string SCARCE_TAG = "scarce";
static const std::string SPARSE_TAG = "sparse";
static const std::string MATRIX_TAG = "matrix";
static const std::string TYPE_TAG = "type";

static const std::string NONE_VALUE("none");

static const std::string NOISE_MODEL_TAG = "noise_model";
static const std::string PRECISION_TAG = "precision";
static const std::string SN_INIT_TAG = "sn_init";
static const std::string SN_MAX_TAG = "sn_max";
static const std::string NOISE_THRESHOLD_TAG = "noise_threshold";

DataConfig::DataConfig ()
   : m_hasData(false) {}

DataConfig::DataConfig ( const Matrix &m
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m);
}

DataConfig::DataConfig ( const SparseMatrix &m
                       , bool isScarce
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m, isScarce);
}

DataConfig::DataConfig ( const DenseTensor &m
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m);
}

DataConfig::DataConfig ( const SparseTensor &m
                       , bool isScarce
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m, isScarce);
}


DataConfig::~DataConfig()
{
}

void DataConfig::check() const
{
   THROWERROR_ASSERT(hasData());

   if (isDense())
   {
      const auto dims = getDims();
      THROWERROR_ASSERT(getNNZ() == std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<std::uint64_t>()));
   }
}

//
// other methods
//

const std::string NON_FINITE_MSG = "Found nan/infinite in input data";

void DataConfig::setData(const Matrix &m)
{
   m_dense_matrix_data = m;
   m_hasData = true;
   m_isDense = true;
   m_isScarce = false;
   m_isMatrix = true;

   THROWERROR_ASSERT_MSG(m.array().isFinite().all(), NON_FINITE_MSG);

   check();
}

void DataConfig::setData(const SparseMatrix &m, bool isScarce)
{
   m_sparse_matrix_data = m;
   m_hasData = true;
   m_isDense = false;
   m_isScarce = isScarce;
   m_isMatrix = true;

   const Eigen::Map<const Vector> values(m.valuePtr(), m.nonZeros());
   THROWERROR_ASSERT_MSG(values.array().isFinite().all(), NON_FINITE_MSG);

   check();
}

void DataConfig::setData(const DenseTensor &m)
{
   m_dense_tensor_data = m;
   m_hasData = true;
   m_isDense = true;
   m_isMatrix = false;

   const Eigen::Map<const Vector> values(m.getValues().data(), m.getValues().size());
   THROWERROR_ASSERT_MSG(values.array().isFinite().all(), NON_FINITE_MSG);;

   check();
}

void DataConfig::setData(const SparseTensor &m, bool isScarce)
{
   m_sparse_tensor_data = m;
   m_hasData = true;
   m_isDense = false;
   m_isScarce = isScarce;
   m_isMatrix = false;

   const Eigen::Map<const Vector> values(m.getValues().data(), m.getValues().size());
   THROWERROR_ASSERT_MSG(values.array().isFinite().all(), NON_FINITE_MSG);;

   check();
}

const Matrix &DataConfig::getDenseMatrixData() const
{
   THROWERROR_ASSERT(hasData() && isDense() && isMatrix());
   return m_dense_matrix_data;
}

const SparseMatrix &DataConfig::getSparseMatrixData() const
{
   THROWERROR_ASSERT(hasData() && !isDense() && isMatrix());
   return m_sparse_matrix_data;
}

const SparseTensor &DataConfig::getSparseTensorData() const
{
   THROWERROR_ASSERT(hasData() && !isDense() && !isMatrix());
   return m_sparse_tensor_data;
}

const DenseTensor &DataConfig::getDenseTensorData() const
{
   THROWERROR_ASSERT(hasData() && isDense() && !isMatrix());
   return m_dense_tensor_data;
}

Matrix &DataConfig::getDenseMatrixData()
{
   THROWERROR_ASSERT(!hasData());
   return m_dense_matrix_data;
}

SparseMatrix &DataConfig::getSparseMatrixData()
{
   THROWERROR_ASSERT(!hasData());
   return m_sparse_matrix_data;
}

SparseTensor &DataConfig::getSparseTensorData()
{
   THROWERROR_ASSERT(!hasData());
   return m_sparse_tensor_data;
}

DenseTensor &DataConfig::getDenseTensorData()
{
   THROWERROR_ASSERT(!hasData());
   return m_dense_tensor_data;
}

bool DataConfig::hasData() const
{
   return m_hasData;
}

bool DataConfig::isDense() const
{
   return m_isDense;
}

bool DataConfig::isScarce() const
{
   return m_isScarce;
}

bool DataConfig::isMatrix() const
{
   return m_isMatrix;
}

std::uint64_t DataConfig::getNNZ() const
{
   THROWERROR_ASSERT(hasData());

        if (isMatrix() && isDense())        return getDenseMatrixData().nonZeros();
   else if (isMatrix() && !isDense())  return getSparseMatrixData().nonZeros();
   else if (!isMatrix() && !isDense()) return getSparseTensorData().getNNZ();
   else if (!isMatrix() && isDense())  return getDenseTensorData().getNNZ();
   else THROWERROR_NOTIMPL();
}

std::uint64_t DataConfig::getNModes() const
{
   return getDims().size();
}

const std::vector<std::uint64_t> DataConfig::getDims() const
{
   THROWERROR_ASSERT(hasData());

   if (isMatrix() && isDense())
   {
      const auto &m = getDenseMatrixData();
      return std::vector<std::uint64_t>{ (std::uint64_t)m.rows(), (std::uint64_t)m.cols() };
   }
   else if (isMatrix() && !isDense())
   {
      const auto &m = getSparseMatrixData();
      return std::vector<std::uint64_t>{ (std::uint64_t)m.rows(), (std::uint64_t)m.cols() };
   }
   else if (!isMatrix() && !isDense())
   {
      const auto &m = getSparseTensorData();
      return m.getDims();
   }
   else if (!isMatrix() && isDense())
   {
      const auto &m = getDenseTensorData();
      return m.getDims();
   }
   else
      THROWERROR_NOTIMPL();
}



const NoiseConfig& DataConfig::getNoiseConfig() const
{
   return m_noiseConfig;
}

void DataConfig::setNoiseConfig(const NoiseConfig& value)
{
   m_noiseConfig = value;
}

void DataConfig::setFilename(const std::string &f)
{
    m_filename = f;
}

const std::string &DataConfig::getFilename() const
{
    return m_filename;
}

void DataConfig::setPos(const PVec<>& p)
{
   m_pos = p;
}

bool DataConfig::hasPos() const
{
    return m_pos.size();
}

const PVec<>& DataConfig::getPos() const
{
   THROWERROR_ASSERT(hasPos());
   return m_pos;
}

std::ostream& DataConfig::info(std::ostream& os) const
{
   if (!hasData())
   {
      os << "0";
   }
   else
   {
      os << getDims().operator[](0);
      for (std::size_t i = 1; i < getDims().size(); i++)
         os << " x " << getDims().operator[](i);
   }
   if (getFilename().size())
   {
        os << " \"" << getFilename() << "\"";
   }
   if (hasPos())
   {
        os << " @[" << getPos() << "]";
   }
   return os;
}

std::string DataConfig::info() const
{
    std::stringstream ss;
    info(ss);
    return ss.str();
}

void DataConfig::save(HDF5Group& cfg_file, const std::string& sectionName) const
{
   if (!hasData())
      return;

   //write tensor config position
   if (hasPos())
   {
      std::stringstream ss;
      ss << getPos();
      cfg_file.put(sectionName, POS_TAG, ss.str());
   }

   if (isMatrix() && isDense())
      cfg_file.write(sectionName, DATA_TAG, getDenseMatrixData());
   else if (isMatrix() && !isDense())
      cfg_file.write(sectionName, DATA_TAG, getSparseMatrixData());
   else if (!isMatrix() && !isDense())
      cfg_file.write(sectionName, DATA_TAG, getSparseTensorData());
   else if (!isMatrix() && isDense())
      cfg_file.write(sectionName, DATA_TAG, getDenseTensorData());
   else
      THROWERROR_NOTIMPL();

   //write tensor config type
   std::string type_str = isDense() ? DENSE_TAG : isScarce() ? SCARCE_TAG : SPARSE_TAG;
   cfg_file.put(sectionName, TYPE_TAG, type_str);
   cfg_file.put(sectionName, MATRIX_TAG, isMatrix());

   //write noise config
   auto &noise_config = getNoiseConfig();
   if (noise_config.getNoiseType() != NoiseTypes::unset)
   {
      cfg_file.put(sectionName, NOISE_MODEL_TAG, noiseTypeToString(noise_config.getNoiseType()));
      cfg_file.put(sectionName, PRECISION_TAG, noise_config.getPrecision());
      cfg_file.put(sectionName, SN_INIT_TAG, noise_config.getSnInit());
      cfg_file.put(sectionName, SN_MAX_TAG, noise_config.getSnMax());
      cfg_file.put(sectionName, NOISE_THRESHOLD_TAG, noise_config.getThreshold());
   }

}

bool DataConfig::restore(const HDF5Group& cfg_file, const std::string& sectionName)
{
   if (!cfg_file.hasSection(sectionName))
      return false;

   //restore position
   std::string pos_str = cfg_file.get(sectionName, POS_TAG, NONE_VALUE);
   if (pos_str != NONE_VALUE)
   {
      std::vector<int> tokens;
      split(pos_str, tokens, ',');

      //assign position
      setPos(PVec<>(tokens));
   }

   //restore type
   m_isDense = cfg_file.get(sectionName, TYPE_TAG, DENSE_TAG) == DENSE_TAG;
   m_isScarce = cfg_file.get(sectionName, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;
   m_isMatrix = cfg_file.get(sectionName, MATRIX_TAG, true);

   if (isMatrix() && isDense())
      cfg_file.read(sectionName, DATA_TAG, getDenseMatrixData());
   else if (isMatrix() && !isDense())
      cfg_file.read(sectionName, DATA_TAG, getSparseMatrixData());
   else if (!isMatrix() && !isDense())
      cfg_file.read(sectionName, DATA_TAG, getSparseTensorData());
   else if (!isMatrix() && isDense())
      cfg_file.read(sectionName, DATA_TAG, getDenseTensorData());
   else
      THROWERROR_NOTIMPL();

   //restore noise model
   NoiseConfig noise;

   NoiseTypes noiseType = stringToNoiseType(cfg_file.get(sectionName, NOISE_MODEL_TAG, noiseTypeToString(NoiseTypes::unset)));
   if (noiseType != NoiseTypes::unset)
   {
      noise.setNoiseType(noiseType);
      noise.setPrecision(cfg_file.get(sectionName, PRECISION_TAG, NoiseConfig::PRECISION_DEFAULT_VALUE));
      noise.setSnInit(cfg_file.get(sectionName, SN_INIT_TAG, NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE));
      noise.setSnMax(cfg_file.get(sectionName, SN_MAX_TAG, NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE));
      noise.setThreshold(cfg_file.get(sectionName, NOISE_THRESHOLD_TAG, NoiseConfig::PROBIT_DEFAULT_VALUE));
   }

   //assign noise model
   setNoiseConfig(noise);

   m_hasData = true;

   check();

   return true;
}

} // end namespace smurff
