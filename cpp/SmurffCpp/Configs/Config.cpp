#include "Config.h"



#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cstdio>
#include <string>
#include <memory>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/HDF5Group.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/StringUtils.h>

namespace smurff {

static const std::string NONE_VALUE = "none";

static const std::string OPTIONS_SECTION_TAG = "options";
static const std::string TRAIN_SECTION_TAG = "train";
static const std::string TEST_SECTION_TAG = "test";

static const std::string NUM_PRIORS_TAG = "num_priors";
static const std::string PRIOR_PREFIX = "prior";
static const std::string NUM_AUX_DATA_TAG = "num_aux_data";
static const std::string AUX_DATA_PREFIX = "aux_data";
static const std::string RESTORE_NAME_TAG = "restore_file";
static const std::string SAVE_NAME_TAG = "save_file";
static const std::string SAVE_FREQ_TAG = "save_freq";
static const std::string SAVE_PRED_TAG = "save_pred";
static const std::string SAVE_MODEL_TAG = "save_model";
static const std::string CHECKPOINT_FREQ_TAG = "checkpoint_freq";
static const std::string VERBOSE_TAG = "verbose";
static const std::string BURNING_TAG = "burnin";
static const std::string NSAMPLES_TAG = "nsamples";
static const std::string NUM_LATENT_TAG = "num_latent";
static const std::string NUM_THREADS_TAG = "num_threads";
static const std::string RANDOM_SEED_SET_TAG = "random_seed_set";
static const std::string RANDOM_SEED_TAG = "random_seed";
static const std::string INIT_MODEL_TAG = "init_model";
static const std::string CLASSIFY_TAG = "classify";
static const std::string THRESHOLD_TAG = "threshold";

static const std::string LAMBDA_TAG = "prop_Lambda";
static const std::string MU_TAG = "prop_mu";

static const std::string PRIOR_NAME_DEFAULT = "default";
static const std::string PRIOR_NAME_MACAU = "macau";
static const std::string PRIOR_NAME_MACAU_ONE = "macauone";
static const std::string PRIOR_NAME_SPIKE_AND_SLAB = "spikeandslab";
static const std::string PRIOR_NAME_NORMAL = "normal";
static const std::string PRIOR_NAME_NORMALONE = "normalone";

static const std::string MODEL_INIT_NAME_RANDOM = "random";
static const std::string MODEL_INIT_NAME_ZERO = "zero";

PriorTypes stringToPriorType(std::string name)
{
   if(name == PRIOR_NAME_DEFAULT)
      return PriorTypes::default_prior;
   else if(name == PRIOR_NAME_MACAU)
      return PriorTypes::macau;
   else if(name == PRIOR_NAME_MACAU_ONE)
      return PriorTypes::macauone;
   else if(name == PRIOR_NAME_SPIKE_AND_SLAB)
      return PriorTypes::spikeandslab;
   else if(name == PRIOR_NAME_NORMALONE)
      return PriorTypes::normalone;
   else if(name == PRIOR_NAME_NORMAL)
      return PriorTypes::normal;
   else
   {
      THROWERROR("Invalid prior type");
   }
}

std::string priorTypeToString(PriorTypes type)
{
   switch(type)
   {
      case PriorTypes::default_prior:
         return PRIOR_NAME_DEFAULT;
      case PriorTypes::macau:
         return PRIOR_NAME_MACAU;
      case PriorTypes::macauone:
         return PRIOR_NAME_MACAU_ONE;
      case PriorTypes::spikeandslab:
         return PRIOR_NAME_SPIKE_AND_SLAB;
      case PriorTypes::normal:
         return PRIOR_NAME_NORMAL;
      case PriorTypes::normalone:
         return PRIOR_NAME_NORMALONE;
      default:
      {
         THROWERROR("Invalid prior type");
      }
   }
}

ModelInitTypes stringToModelInitType(std::string name)
{
   if(name == MODEL_INIT_NAME_RANDOM)
      return ModelInitTypes::random;
   else if (name == MODEL_INIT_NAME_ZERO)
      return ModelInitTypes::zero;
   else
   {
      THROWERROR("Invalid model init type " + name);
   }
}

std::string modelInitTypeToString(ModelInitTypes type)
{
   switch(type)
   {
      case ModelInitTypes::random:
         return MODEL_INIT_NAME_RANDOM;
      case ModelInitTypes::zero:
         return MODEL_INIT_NAME_ZERO;
      default:
      {
         THROWERROR("Invalid model init type");
      }
   }
}

//config
int Config::BURNIN_DEFAULT_VALUE = 200;
int Config::NSAMPLES_DEFAULT_VALUE = 800;
int Config::NUM_LATENT_DEFAULT_VALUE = 32;
int Config::NUM_THREADS_DEFAULT_VALUE = 0; // as many as you want
ModelInitTypes Config::INIT_MODEL_DEFAULT_VALUE = ModelInitTypes::zero;
std::string Config::SAVE_NAME_DEFAULT_VALUE = std::string();
int Config::SAVE_FREQ_DEFAULT_VALUE = 0;
bool Config::SAVE_PRED_DEFAULT_VALUE = true;
bool Config::SAVE_MODEL_DEFAULT_VALUE = true;
int Config::CHECKPOINT_FREQ_DEFAULT_VALUE = 0;
int Config::VERBOSE_DEFAULT_VALUE = 0;
const std::string Config::STATUS_DEFAULT_VALUE = "";
bool Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE = true;
double Config::THRESHOLD_DEFAULT_VALUE = 0.0;
int Config::RANDOM_SEED_DEFAULT_VALUE = 0;

Config::Config()
{
   m_model_init_type = Config::INIT_MODEL_DEFAULT_VALUE;

   m_restore_name.clear();

   m_save_name = Config::SAVE_NAME_DEFAULT_VALUE;
   m_save_freq = Config::SAVE_FREQ_DEFAULT_VALUE;
   m_save_pred = Config::SAVE_PRED_DEFAULT_VALUE;
   m_save_model = Config::SAVE_MODEL_DEFAULT_VALUE;
   m_checkpoint_freq = Config::CHECKPOINT_FREQ_DEFAULT_VALUE;

   m_random_seed_set = false;
   m_random_seed = Config::RANDOM_SEED_DEFAULT_VALUE;

   m_verbose = Config::VERBOSE_DEFAULT_VALUE;
   m_burnin = Config::BURNIN_DEFAULT_VALUE;
   m_nsamples = Config::NSAMPLES_DEFAULT_VALUE;
   m_num_latent = Config::NUM_LATENT_DEFAULT_VALUE;
   m_num_threads = Config::NUM_THREADS_DEFAULT_VALUE;

   m_threshold = Config::THRESHOLD_DEFAULT_VALUE;
   m_classify = false;
}

const SideInfoConfig& Config::getSideInfoConfig(int mode) const
{
  auto iter = m_sideInfoConfigs.find(mode);
  THROWERROR_ASSERT(iter != m_sideInfoConfigs.end());
  return iter->second;
}

SideInfoConfig& Config::addSideInfo(int mode, const SideInfoConfig &c)
{
   THROWERROR_ASSERT(!hasSideInfo(mode));

   // automagically update prior type
   // normal(one) prior -> macau(one) prior
   if ((int)m_prior_types.size() > mode)
   {
      PriorTypes &pt = m_prior_types[mode];
      if (pt == PriorTypes::normal)
         pt = PriorTypes::macau;
      else if (pt == PriorTypes::normalone)
         pt = PriorTypes::macauone;
   }

   auto p = m_sideInfoConfigs.insert(std::make_pair(mode, c));
   THROWERROR_ASSERT(p.second);

   return p.first->second;
}

bool Config::validate() const
{
   if (!getTrain().hasData() || getTrain().getNNZ() == 0)
   {
      THROWERROR("Missing train data");
   }

   THROWERROR_ASSERT(getTrain().hasPos());

   if (getTest().hasData() && m_test.getNNZ() == 0)
   {
      THROWERROR("Empty test data matrix/tensor");
   }

   if (getTest().hasData() && getTest().getDims() != getTrain().getDims())
   {
      THROWERROR("Train and test data should have the same dimensions");
   }

   if(getPriorTypes().size() != getTrain().getNModes())
   {
      THROWERROR("Number of priors should equal to number of dimensions in train data");
   }

   if ((getTrain().getNModes() > 2) && (getData().size() > 1))
   {
      //it is advised to check macau and macauone priors implementation
      //as well as code in PriorFactory that creates macau priors

      //this check does not directly check that input data is DenseTensor (it only checks number of dimensions)
      //however TensorDataFactory will do an additional check throwing an exception
      THROWERROR("Aux data is not supported for TensorData");
   }

   for (auto p : m_sideInfoConfigs)
   {
      int mode = p.first;
      auto &configItem = p.second;

      if (configItem.getDims()[0] != getTrain().getDims()[mode])
      {
         std::stringstream ss;
         ss << "Side info should have the same number of rows as size of dimension " << mode << " in train data";
         THROWERROR(ss.str());
      }
   }

   for(auto ad1 = m_data.begin(); ad1 != m_data.end(); ad1++)
   {
      if (!ad1->hasPos())
      {
         std::stringstream ss;
         ss << "Data \"" << ad1->info() << "\" is missing position info";
         THROWERROR(ss.str());
      }

      const auto& dim1 = ad1->getDims();
      const auto& pos1 = ad1->getPos();

      auto ad2 = ad1;
      for(ad2++; ad2 != m_data.end(); ad2++)
      {
         const auto& dim2 = ad2->getDims();
         const auto& pos2 = ad2->getPos();

         if (pos1 == pos2)
         {
            std::stringstream ss;
            ss << "Data \"" << ad1->info() <<  "\" and \"" << ad2->info() << "\" at same position";
            THROWERROR(ss.str());
         }

         // if two data blocks are aligned in a certain dimension
         // this dimension should be equal size
         for (std::size_t i = 0; i < pos1.size(); ++i)
         {
            if (pos1.at(i) == pos2.at(i) && (dim1.at(i) != dim2.at(i)))
            {
               std::stringstream ss;
               ss << "Data \"" << ad1->info() << "\" and \"" << ad2->info() << "\" different in size in dimension " << i;
               THROWERROR(ss.str());
            }
         }
      }
   }

   for(std::size_t i = 0; i < m_prior_types.size(); i++)
   {
      PriorTypes pt = m_prior_types[i];
      switch (pt)
      {
         case PriorTypes::normal:
         case PriorTypes::normalone:
         case PriorTypes::spikeandslab:
         case PriorTypes::default_prior:
            THROWERROR_ASSERT_MSG(!hasSideInfo(i), priorTypeToString(pt) + " prior in dimension " + std::to_string(i) + " cannot have side info");
            break;
         case PriorTypes::macau:
         case PriorTypes::macauone:
            THROWERROR_ASSERT_MSG(hasSideInfo(i), priorTypeToString(pt) + " prior in dimension " + std::to_string(i) + " needs side info");
            break;
         default:
            THROWERROR("Unknown prior");
            break;
      }
   }

   getTrain().getNoiseConfig().validate();

   // validate propagated posterior
   for(uint64_t i=0; i<getTrain().getNModes(); ++i)
   {
       if (hasPropagatedPosterior(i))
       {
           THROWERROR_ASSERT_MSG(
               getMuPropagatedPosterior(i).getNCol() == getTrain().getDims().at(i),
               "mu of propagated posterior in mode " + std::to_string(i) + 
               " should have same number of columns as train in mode"
           );
           THROWERROR_ASSERT_MSG(
               getLambdaPropagatedPosterior(i).getNCol() == getTrain().getDims().at(i),
               "Lambda of propagated posterior in mode " + std::to_string(i) + 
               " should have same number of columns as train in mode"
           );
           THROWERROR_ASSERT_MSG(
               (int)getMuPropagatedPosterior(i).getNRow() == getNumLatent(),
               "mu of propagated posterior in mode " + std::to_string(i) + 
               " should have num-latent rows"
           );
           THROWERROR_ASSERT_MSG(
               (int)getLambdaPropagatedPosterior(i).getNRow() == getNumLatent() * getNumLatent(),
               "mu of propagated posterior in mode " + std::to_string(i) +
                   " should have num-latent^2 rows"
           );
       }
   }

   return true;
}

HDF5Group &Config::save(HDF5Group &cfg_file) const
{
   //count data
   cfg_file.put(OPTIONS_SECTION_TAG, NUM_PRIORS_TAG, m_prior_types.size());
   cfg_file.put(OPTIONS_SECTION_TAG, NUM_AUX_DATA_TAG, getData().size() - 1);

   //priors data
   int prior_idx = 0;
   for(const auto &pt : m_prior_types)
      cfg_file.put(OPTIONS_SECTION_TAG, addIndex(PRIOR_PREFIX, prior_idx++), priorTypeToString(pt));

   //save data
   cfg_file.put(OPTIONS_SECTION_TAG, RESTORE_NAME_TAG, m_restore_name);
   cfg_file.put(OPTIONS_SECTION_TAG, SAVE_NAME_TAG, m_save_name);
   cfg_file.put(OPTIONS_SECTION_TAG, SAVE_FREQ_TAG, m_save_freq);
   cfg_file.put(OPTIONS_SECTION_TAG, SAVE_PRED_TAG, m_save_pred);
   cfg_file.put(OPTIONS_SECTION_TAG, SAVE_MODEL_TAG, m_save_model);
   cfg_file.put(OPTIONS_SECTION_TAG, CHECKPOINT_FREQ_TAG, m_checkpoint_freq);

   //general data
   cfg_file.put(OPTIONS_SECTION_TAG, VERBOSE_TAG, m_verbose);
   cfg_file.put(OPTIONS_SECTION_TAG, BURNING_TAG, m_burnin);
   cfg_file.put(OPTIONS_SECTION_TAG, NSAMPLES_TAG, m_nsamples);
   cfg_file.put(OPTIONS_SECTION_TAG, NUM_LATENT_TAG, m_num_latent);
   cfg_file.put(OPTIONS_SECTION_TAG, NUM_THREADS_TAG, m_num_threads);
   cfg_file.put(OPTIONS_SECTION_TAG, RANDOM_SEED_SET_TAG, m_random_seed_set);
   cfg_file.put(OPTIONS_SECTION_TAG, RANDOM_SEED_TAG, m_random_seed);
   cfg_file.put(OPTIONS_SECTION_TAG, INIT_MODEL_TAG, modelInitTypeToString(m_model_init_type));

   //probit prior data
   cfg_file.put(OPTIONS_SECTION_TAG, CLASSIFY_TAG, m_classify);
   cfg_file.put(OPTIONS_SECTION_TAG, THRESHOLD_TAG, m_threshold);

   //write train data section
   getTrain().save(cfg_file, TRAIN_SECTION_TAG);

   //write test data section
   getTest().save(cfg_file, TEST_SECTION_TAG);

   //write macau prior configs section
   for (auto p : m_sideInfoConfigs)
   {
       int mode = p.first;
       auto &configItem = p.second;
       configItem.save(cfg_file, mode);
   }

   //write data section -- excluding train
   for (std::size_t sIndex = 1; sIndex < m_data.size(); sIndex++)
      m_data.at(sIndex).save(cfg_file, addIndex(AUX_DATA_PREFIX, sIndex-1));

   //write posterior propagation
   for (std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
   {
       if (hasPropagatedPosterior(pIndex))
       {
           getMuPropagatedPosterior(pIndex).save(cfg_file, addIndex(MU_TAG, pIndex));
           getLambdaPropagatedPosterior(pIndex).save(cfg_file, addIndex(LAMBDA_TAG, pIndex));
       }
   }

   return cfg_file;
}

bool Config::restore(const HDF5Group &cfg_file)
{

   //restore train data
   getTest().restore(cfg_file, TEST_SECTION_TAG);

   //restore test data
   getTrain().restore(cfg_file, TRAIN_SECTION_TAG);

   //restore priors
   std::size_t num_priors = cfg_file.get(OPTIONS_SECTION_TAG, NUM_PRIORS_TAG, 0);
   std::vector<std::string> pNames;
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      pNames.push_back(cfg_file.get(OPTIONS_SECTION_TAG, addIndex(PRIOR_PREFIX, pIndex),  PRIOR_NAME_DEFAULT));
   }
   setPriorTypes(pNames);

   //restore macau prior configs section
   for (std::size_t mPriorIndex = 0; mPriorIndex < num_priors; mPriorIndex++)
   {
      SideInfoConfig sideInfoConfig;
      if (sideInfoConfig.restore(cfg_file, mPriorIndex))
         m_sideInfoConfigs[mPriorIndex] = sideInfoConfig;
   }

   //restore aux data
   std::size_t num_aux_data = cfg_file.get(OPTIONS_SECTION_TAG, NUM_AUX_DATA_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_aux_data; pIndex++)
   {
      addData().restore(cfg_file, addIndex(AUX_DATA_PREFIX, pIndex));
   }

   // restore posterior propagated data
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      getMuPropagatedPosterior(pIndex).restore(cfg_file, addIndex(MU_TAG, pIndex));
      getLambdaPropagatedPosterior(pIndex).restore(cfg_file, addIndex(LAMBDA_TAG, pIndex));
   }


   //restore save data
   m_restore_name = cfg_file.get(OPTIONS_SECTION_TAG, RESTORE_NAME_TAG, std::string());
   m_save_name  = cfg_file.get(OPTIONS_SECTION_TAG, SAVE_NAME_TAG, std::string());
   m_save_freq  = cfg_file.get(OPTIONS_SECTION_TAG, SAVE_FREQ_TAG, Config::SAVE_FREQ_DEFAULT_VALUE);
   m_save_pred  = cfg_file.get(OPTIONS_SECTION_TAG, SAVE_PRED_TAG, Config::SAVE_PRED_DEFAULT_VALUE);
   m_save_model = cfg_file.get(OPTIONS_SECTION_TAG, SAVE_MODEL_TAG, Config::SAVE_MODEL_DEFAULT_VALUE);
   m_checkpoint_freq = cfg_file.get(OPTIONS_SECTION_TAG, CHECKPOINT_FREQ_TAG, Config::CHECKPOINT_FREQ_DEFAULT_VALUE);

   //restore general data
   m_verbose = cfg_file.get(OPTIONS_SECTION_TAG, VERBOSE_TAG, Config::VERBOSE_DEFAULT_VALUE);
   m_burnin = cfg_file.get(OPTIONS_SECTION_TAG, BURNING_TAG, Config::BURNIN_DEFAULT_VALUE);
   m_nsamples = cfg_file.get(OPTIONS_SECTION_TAG, NSAMPLES_TAG, Config::NSAMPLES_DEFAULT_VALUE);
   m_num_latent = cfg_file.get(OPTIONS_SECTION_TAG, NUM_LATENT_TAG, Config::NUM_LATENT_DEFAULT_VALUE);
   m_num_threads = cfg_file.get(OPTIONS_SECTION_TAG, NUM_THREADS_TAG, Config::NUM_THREADS_DEFAULT_VALUE);
   m_random_seed_set = cfg_file.get(OPTIONS_SECTION_TAG, RANDOM_SEED_SET_TAG,  false);
   m_random_seed = cfg_file.get(OPTIONS_SECTION_TAG, RANDOM_SEED_TAG, Config::RANDOM_SEED_DEFAULT_VALUE);
   m_model_init_type = stringToModelInitType(cfg_file.get(OPTIONS_SECTION_TAG, INIT_MODEL_TAG, modelInitTypeToString(Config::INIT_MODEL_DEFAULT_VALUE)));

   //restore probit prior data
   m_classify = cfg_file.get(OPTIONS_SECTION_TAG, CLASSIFY_TAG,  false);
   m_threshold = cfg_file.get(OPTIONS_SECTION_TAG, THRESHOLD_TAG, Config::THRESHOLD_DEFAULT_VALUE);

   return true;
}

std::ostream& Config::info(std::ostream &os, std::string indent) const
{
   os << indent << "  Iterations: " << getBurnin() << " burnin + " << getNSamples() << " samples\n";

   if (getSaveFreq() != 0 || getCheckpointFreq() != 0)
   {
      if (getSaveFreq() > 0)
      {
          os << indent << "  Save model: every " << getSaveFreq() << " iteration\n";
      }
      else if (getSaveFreq() < 0)
      {
          os << indent << "  Save model after last iteration\n";
      }

      if (getCheckpointFreq() > 0)
      {
          os << indent << "  Checkpoint state: every " << getCheckpointFreq() << " seconds\n";
      }

      os << indent << "  Output file: " << getSaveName() << "\n";
   }
   else
   {
      os << indent << "  Save model: never\n";
   }


   if (!getRestoreName().empty())
      os << indent << "  Input file: " << getRestoreName() << "\n";

   return os;
}

} // end namespace smurff
