#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Utils/Error.h>
#include "DataConfig.h"
#include "SideInfoConfig.h"

namespace smurff {

class HDF5Group;

enum class PriorTypes
{
   default_prior,
   macau,
   macauone,
   spikeandslab,
   normal,
   normalone,
   mpi
};

enum class ModelInitTypes
{
   random,
   zero
};


PriorTypes stringToPriorType(std::string name);

std::string priorTypeToString(PriorTypes type);

ModelInitTypes stringToModelInitType(std::string name);

std::string modelInitTypeToString(ModelInitTypes type);

struct Config
{
public:

   //config
   static int BURNIN_DEFAULT_VALUE;
   static int NSAMPLES_DEFAULT_VALUE;
   static int NUM_LATENT_DEFAULT_VALUE;
   static int NUM_THREADS_DEFAULT_VALUE;
   static bool POSTPROP_DEFAULT_VALUE;
   static ModelInitTypes INIT_MODEL_DEFAULT_VALUE;
   static std::string SAVE_NAME_DEFAULT_VALUE;
   static int SAVE_FREQ_DEFAULT_VALUE;
   static bool SAVE_PRED_DEFAULT_VALUE;
   static bool SAVE_MODEL_DEFAULT_VALUE;
   static int CHECKPOINT_FREQ_DEFAULT_VALUE;
   static int VERBOSE_DEFAULT_VALUE;
   static const std::string STATUS_DEFAULT_VALUE;
   static bool ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
   static double THRESHOLD_DEFAULT_VALUE;
   static int RANDOM_SEED_DEFAULT_VALUE;

private:
   //-- train and test
   DataConfig m_test;
   DataConfig m_row_features;
   DataConfig m_col_features;

   //-- all train data (including train and aux)
   std::vector<DataConfig> m_data;

   //-- sideinfo per mode
   std::map<int, SideInfoConfig> m_sideInfoConfigs;

   // -- priors
   std::vector<PriorTypes> m_prior_types;

   // -- posterior propagation
   std::map<int, DataConfig> m_mu_postprop;
   std::map<int, DataConfig> m_lambda_postprop;

   //-- init model
   ModelInitTypes m_model_init_type;

   //-- save
   int m_save_freq;
   bool m_save_pred;
   bool m_save_model;
   int m_checkpoint_freq;

   //-- general
   bool m_random_seed_set;
   int m_random_seed;
   int m_verbose;
   int m_burnin;
   int m_nsamples;
   int m_num_latent;
   int m_num_threads; 

   //-- binary classification
   bool m_classify;
   double m_threshold;

   //-- meta
   std::string m_restore_name;
   std::string m_save_name;
   std::string m_ini_name;

   const PVec<> getTrainPos() const
   {
      return PVec<>(getNModes());
   }

 public:
   Config();

public:
   bool validate() const;

   HDF5Group &save(HDF5Group &) const;
   bool restore(const HDF5Group &);

   std::ostream& info(std::ostream &os, std::string indent) const;

public:
   bool isActionTrain()
   {
       return getTrain().hasData();
   }

   bool isActionPredict()
   {
       return getPredict().hasData() || getColFeatures().hasData() || getRowFeatures().hasData();
   }

   const DataConfig& getTrain() const
   {
      return m_data.at(0);
   }

   DataConfig& getTrain() 
   {
      if (m_data.size() == 0) m_data.resize(1);

      auto &train_config = m_data.at(0);

      if (!train_config.hasPos()) 
         train_config.setPos(PVec<>{0,0});
      else 
         THROWERROR_ASSERT((train_config.getPos().as_vector() == std::vector<std::int64_t>{0,0}));

      return train_config;
   }
   
   const DataConfig &getTest() const
   {
      return m_test;
   }

   DataConfig &getTest() 
   {
      return m_test;
   }
   
   const DataConfig &getRowFeatures() const
   {
      return m_row_features;
   }


   const DataConfig &getColFeatures() const
   {
      return m_col_features;
   }

   DataConfig &getRowFeatures()
   {
      return m_row_features;
   }

   DataConfig &getColFeatures() 
   {
      return m_col_features;
   }

   DataConfig &getPredict()
   {
      return m_test;
   }

   const std::map<int, SideInfoConfig>& getSideInfoConfigs() const
   {
      return m_sideInfoConfigs;
   }

   const SideInfoConfig& getSideInfoConfig(int mode) const;

   SideInfoConfig& addSideInfo(int mode, const SideInfoConfig & = SideInfoConfig());

   bool hasSideInfo(int mode) const
   {
       return m_sideInfoConfigs.find(mode) != m_sideInfoConfigs.end();
   }

   const std::vector<DataConfig> &getData() const
   {
      return m_data;
   }
   
   DataConfig &addData(const DataConfig & = DataConfig())
   {
      m_data.push_back(DataConfig());
      return m_data.back();
   }

   unsigned getNModes() const
   {
      THROWERROR_ASSERT(!m_prior_types.empty());
      return m_prior_types.size();
   }

   const std::vector<PriorTypes> getPriorTypes() const
   {
      return m_prior_types;
   }

   void setPriorTypes(std::vector<PriorTypes> values)
   {
      m_prior_types = values;
   }

   void setPriorTypes(std::vector<std::string> values)
   {
      m_prior_types.clear();
      for(auto &value : values)
          m_prior_types.push_back(stringToPriorType(value));
   }

   bool hasPropagatedPosterior(int mode) const
   {
       return m_mu_postprop.find(mode) != m_mu_postprop.end() && getMuPropagatedPosterior(mode).hasData();
   }

   void addPropagatedPosterior(int mode, const Matrix &mu, const Matrix &Lambda) 
   {
      getMuPropagatedPosterior(mode).setData(mu);
      getLambdaPropagatedPosterior(mode).setData(Lambda);
   }

   const DataConfig& getMuPropagatedPosterior(int mode) const
   {
       return m_mu_postprop.find(mode)->second;
   }

   const DataConfig& getLambdaPropagatedPosterior(int mode) const
   {
       return m_lambda_postprop.find(mode)->second;
   }

   DataConfig& getMuPropagatedPosterior(int mode) 
   {
       return m_mu_postprop[mode];
   }

   DataConfig& getLambdaPropagatedPosterior(int mode) 
   {
       return m_lambda_postprop[mode];
   }

   ModelInitTypes getModelInitType() const
   {
      return m_model_init_type;
   }

   void setModelInitType(ModelInitTypes value)
   {
      m_model_init_type = value;
   }

   std::string getModelInitTypeAsString() const
   {
      return modelInitTypeToString(m_model_init_type);
   }

   void setModelInitType(std::string value)
   {
      m_model_init_type = stringToModelInitType(value);
   }

   std::string getRestoreName() const 
   {
      return m_restore_name;
   }

   void setRestoreName(std::string value)
   {
      m_restore_name = value;
   }

   std::string getSaveName() const 
   {
      return m_save_name;
   }

   void setSaveName(std::string value)
   {
      m_save_name = value;
   }

   int getSaveFreq() const
   {
      return m_save_freq;
   }

   void setSaveFreq(int value)
   {
      m_save_freq = value;
   }

   bool getSavePred() const
   {
      return m_save_pred;
   }

   void setSavePred(bool value)
   {
      m_save_pred = value;
   }

   bool getSaveModel() const
   {
      return m_save_model;
   }

   void setSaveModel(bool value)
   {
      m_save_model = value;
   }

   int getCheckpointFreq() const
   {
      return m_checkpoint_freq;
   }

   void setCheckpointFreq(int value)
   {
      m_checkpoint_freq = value;
   }

   bool getRandomSeedSet() const
   {
      return m_random_seed_set;
   }

   int getRandomSeed() const
   {
      THROWERROR_ASSERT_MSG(getRandomSeedSet(), "Random seed is unset");
      return m_random_seed;
   }

   void setRandomSeed(int value)
   {
      m_random_seed_set = true;
      m_random_seed = value;
   }

   int getVerbose() const
   {
      return m_verbose;
   }

   void setVerbose(int value)
   {
      if (value < 0) value = 0;
      m_verbose = value;
   }

   int getBurnin() const
   {
      return m_burnin;
   }

   void setBurnin(int value)
   {
      m_burnin = value;
   }

   int getNSamples() const
   {
      return m_nsamples;
   }

   void setNSamples(int value)
   {
      m_nsamples = value;
   }

   int getNumLatent() const
   {
      return m_num_latent;
   }

   void setNumLatent(int value)
   {
      m_num_latent = value;
   }

   bool getClassify() const
   {
      return m_classify;
   }

   double getThreshold() const
   {
      return m_threshold;
   }

   void setThreshold(double value)
   {
      m_threshold = value;
      m_classify = true;
   }

   int getNumThreads() const
   {
       return m_num_threads;
   }

   void setNumThreads(int value)
   {
       m_num_threads = value;
   }

   std::string getIniName() const
   {
       return m_ini_name;
   }

   void setIniName(std::string value)
   {
       m_ini_name = value;
   } 
};

}

