#include "DataCreator.h"

#include "DataCreatorBase.h"

#include <SmurffCpp/DataMatrices/MatricesData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

#include <SmurffCpp/Configs/DataConfig.h>


#include <Utils/Error.h>
#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

std::shared_ptr<Data> DataCreator::create(const DataConfig& dc) const
{
   auto& aux_matrices = getSession().getConfig().getAuxData();

   //create creator
   std::shared_ptr<DataCreatorBase> creatorBase = std::make_shared<DataCreatorBase>();

   //create single matrix
   if (aux_matrices.empty())
      return dc.create(creatorBase);

   if (dc.isMatrix())
   {
      //multiple matrices
      NoiseConfig ncfg(NoiseTypes::unused);
      std::shared_ptr<MatricesData> local_data_ptr(new MatricesData());
      local_data_ptr->setNoiseModel(NoiseFactory::create_noise_model(ncfg));
      local_data_ptr->add(PVec<>({0,0}), dc.create(creatorBase));

      for(auto &m : aux_matrices)
      {
         local_data_ptr->add(m->getPos(), m->create(creatorBase));
      }

      return local_data_ptr;
   }
   else
   {
      THROWERROR("Tensor config does not support aux data");
   }

   return std::shared_ptr<Data>();
   
}
} // end namespace smurff
