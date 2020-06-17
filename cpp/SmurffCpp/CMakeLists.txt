FILE (GLOB UTIL_FILES   
                        "Utils/Distribution.h"
                        "Utils/MatrixUtils.h"
                        "Utils/PVec.hpp"
                        "Utils/StateFile.h"
                        "Utils/SaveState.h"
                        "Utils/HDF5Group.h"
                        "Utils/TruncNorm.h"
                        "Utils/InvNormCdf.h"
                        "Utils/counters.h"
                        "Utils/linop.h"
                        "Utils/omp_util.h"
                        "Utils/Error.h"
                        "Utils/ThreadVector.hpp"
                        "Utils/StringUtils.h"
                        "Utils/Tensor.h"
                        "Utils/Distribution.cpp"
                        "Utils/MatrixUtils.cpp"
                        "Utils/StateFile.cpp"
                        "Utils/SaveState.cpp"
                        "Utils/HDF5Group.cpp"
                        "Utils/TruncNorm.cpp"
                        "Utils/InvNormCdf.cpp"
                        "Utils/counters.cpp"
                        "Utils/omp_util.cpp"
                        "Utils/StringUtils.cpp"
                        "Utils/Tensor.cpp"
                        )

source_group ("Utils" FILES ${UTIL_FILES})

FILE (GLOB CONFIG_FILES "Configs/Config.h"
                        "Configs/DataConfig.h"
                        "Configs/NoiseConfig.h"
                        "Configs/SideInfoConfig.h"

                        "Configs/Config.cpp"
                        "Configs/DataConfig.cpp"
                        "Configs/NoiseConfig.cpp"
                        "Configs/SideInfoConfig.cpp"
                        )

source_group ("Configs" FILES ${CONFIG_FILES})

FILE (GLOB NOISE_FILES  "Noises/GaussianNoise.h"
                        "Noises/AdaptiveGaussianNoise.h"
                        "Noises/FixedGaussianNoise.h"
                        "Noises/SampledGaussianNoise.h"
                        "Noises/ProbitNoise.h"
                        "Noises/UnusedNoise.h"
                        "Noises/INoiseModel.h"
                        "Noises/NoiseFactory.h"

                        "Noises/INoiseModel.cpp"
                        "Noises/GaussianNoise.cpp"
                        "Noises/AdaptiveGaussianNoise.cpp"
                        "Noises/FixedGaussianNoise.cpp"
                        "Noises/SampledGaussianNoise.cpp"
                        "Noises/ProbitNoise.cpp"
                        "Noises/UnusedNoise.cpp"
                        "Noises/NoiseFactory.cpp"
                        )

source_group ("Noises" FILES ${NOISE_FILES})

FILE (GLOB PRIOR_FILES  "Priors/ILatentPrior.h"
                        "Priors/NormalPrior.h"
                        "Priors/NormalOnePrior.h"
                        "Priors/SpikeAndSlabPrior.h"
                        "Priors/MacauPrior.h"
                        "Priors/MacauOnePrior.h"
                        "Priors/IPriorFactory.h"
                        "Priors/PriorFactory.h"

                        "Priors/ILatentPrior.cpp"
                        "Priors/NormalPrior.cpp"
                        "Priors/NormalOnePrior.cpp"
                        "Priors/SpikeAndSlabPrior.cpp"
                        "Priors/MacauPrior.cpp"
                        "Priors/MacauOnePrior.cpp"
                        "Priors/PriorFactory.cpp"
                        )

source_group ("Priors" FILES ${PRIOR_FILES})

FILE (GLOB MATRIX_FILES "DataMatrices/Data.h"
                        "DataMatrices/DenseMatrixData.h"
                        "DataMatrices/FullMatrixData.hpp"
                        "DataMatrices/MatricesData.h"
                        "DataMatrices/MatrixData.h"
                        "DataMatrices/MatrixDataTempl.hpp"
                        "DataMatrices/ScarceMatrixData.h"
                        "DataMatrices/SparseMatrixData.h"

                        "DataMatrices/Data.cpp"
                        "DataMatrices/DenseMatrixData.cpp"
                        "DataMatrices/MatricesData.cpp"
                        "DataMatrices/MatrixData.cpp"
                        "DataMatrices/ScarceMatrixData.cpp"
                        "DataMatrices/SparseMatrixData.cpp"
                        )

source_group ("DataMatrices" FILES ${MATRIX_FILES})

FILE (GLOB TENSOR_FILES "DataTensors/TensorData.h"
                        "DataTensors/SparseMode.h"
                        "DataTensors/TensorData.cpp"
                        "DataTensors/SparseMode.cpp"
                        )

source_group ("DataTensors" FILES ${TENSOR_FILES})

FILE (GLOB SESSION_FILES "Sessions/ISession.h"
                         "Sessions/TrainSession.h"
                         "Sessions/CmdSession.h"
                         "Sessions/PythonSession.h"

                         "Sessions/ISession.cpp"
                         "Sessions/TrainSession.cpp"
                         "Sessions/CmdSession.cpp"
                         "Sessions/PythonSession.cpp"
                         )

source_group ("Sessions" FILES ${SESSION_FILES})

FILE (GLOB HEADER_FILES "Model.h"
                        "result.h"
                        "StatusItem.h"
                        "VMatrixIterator.hpp"
                        "ConstVMatrixIterator.hpp"
                        "VMatrixExprIterator.hpp"
                        "ConstVMatrixExprIterator.hpp"
                        )

source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "Model.cpp"
                        "result.cpp"
                        "StatusItem.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

FILE (GLOB SIDE_INFO_FILES "SideInfo/ISideInfo.h"
                           "SideInfo/DenseDoubleFeat.h"
                           "SideInfo/DenseSideInfo.h"
                           "SideInfo/DenseSideInfo.cpp"
                           "SideInfo/SparseSideInfo.h"
                           "SideInfo/SparseSideInfo.cpp"
                           "SideInfo/linop.h"
                           "SideInfo/linop.cpp"
                        )
source_group ("Side Info" FILES ${SIDE_INFO_FILES})

FILE (GLOB PREDICT_FILES "Predict/PredictSession.h"
                         "Predict/PredictSession.cpp"
                        )
                        
source_group ("Prediction" FILES ${PREDICT_FILES})

set(SMURFF_CPP_FILES ${HEADER_FILES}
                     ${SOURCE_FILES}
                     ${UTIL_FILES}
                     ${IO_FILES}
                     ${CONFIG_FILES}
                     ${NOISE_FILES}
                     ${PRIOR_FILES}
                     ${MATRIX_FILES}
                     ${TENSOR_FILES}
                     ${SESSION_FILES}
                     ${SIDE_INFO_FILES}
                     ${PREDICT_FILES})

#SETUP OUTPUT

add_library (smurff-cpp STATIC ${SMURFF_CPP_FILES})
target_link_libraries(smurff-cpp HighFive ${Boost_LIBRARIES})

