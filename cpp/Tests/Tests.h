#include <vector>

#include <SmurffCpp/Types.h>

namespace smurff
{

struct ResultItem;
class DataConfig;

namespace test
{

const double rmse_epsilon = smurff::approx_epsilon<smurff::float_type>();
const double single_item_epsilon = rmse_epsilon * 10;

// noise
extern smurff::NoiseConfig fixed_ncfg;

// dense train data
extern smurff::Matrix trainDenseMatrix;
extern smurff::DenseTensor trainDenseTensor2d;
extern smurff::DenseTensor trainDenseTensor3d;

// sparse train data
extern smurff::SparseMatrix trainSparseMatrix;
extern smurff::SparseTensor trainSparseTensor2d;
extern smurff::SparseTensor trainSparseTensor3d;

// sparse test data
extern smurff::SparseMatrix testSparseMatrix;
extern smurff::SparseTensor testSparseTensor2d;
extern smurff::SparseTensor testSparseTensor3d;

// aux data
extern smurff::DataConfig rowAuxDense;
extern smurff::DataConfig colAuxDense;

// side info
extern smurff::Matrix rowSideDenseMatrix;
extern smurff::Matrix colSideDenseMatrix;
extern smurff::Matrix rowSideDenseMatrix3d;

extern smurff::SparseMatrix rowSideSparseMatrix;
extern smurff::SparseMatrix colSideSparseMatrix;

void checkResultItems(const std::vector<smurff::ResultItem> &actualResultItems,
                          const std::vector<smurff::ResultItem> &expectedResultItems);

template <class M>
SideInfoConfig makeSideInfoConfig(const M &data, bool direct = true)
{
  smurff::NoiseConfig sampled_ncfg(NoiseTypes::sampled);
  sampled_ncfg.setPrecision(10.0);
  SideInfoConfig picfg(data, sampled_ncfg);
  picfg.setDirect(direct);
  return picfg;
}

template <class Train, class Test>
Config genConfig(const Train &train, const Test &test, std::vector<PriorTypes> priors)
{
  Config config;
  config.setPriorTypes(priors);
  config.setBurnin(50);
  config.setNSamples(50);
  config.setVerbose(0);
  config.setRandomSeed(1234);
  config.setNumLatent(4);

  config.getTrain().setData(train);
  config.getTrain().setNoiseConfig(fixed_ncfg);
  config.getTest().setData(test);

  return config;
}

void checkValue(double actualValue, double expectedValue, double epsilon);

} // namespace test
} // namespace smurff
