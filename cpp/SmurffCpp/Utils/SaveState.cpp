#include <iostream>

#include <SmurffCpp/Utils/SaveState.h>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>


#define LATENTS_SEC_TAG "latents"
#define PRED_SEC_TAG "predictions"
#define LINK_MATRICES_SEC_TAG "link_matrices"

#define IS_CHECKPOINT_TAG "is_checkpoint"
#define NUMBER_TAG "number"
#define NUM_MODES_TAG "num_modes"
#define PRED_TAG "pred"
#define PRED_STATE_TAG "pred_state"
#define PRED_AVG_TAG "pred_avg"
#define PRED_VAR_TAG "pred_var"
#define PRED_ONE_TAG "pred_1sample"

#define RMSE_AVG_TAG "rmse_avg"
#define RMSE_1SAMPLE_TAG "rmse_1sample"
#define AUC_AVG_TAG "auc_avg"
#define AUC_1SAMPLE_TAG "auc_1sample"
#define SAMPLE_ITER_TAG "sample_iter"
#define BURNIN_ITER_TAG "burnin_iter"

#define LATENTS_PREFIX "latents_"
#define LINK_MATRIX_PREFIX "link_matrix_"
#define MU_PREFIX "mu_"

#define POST_NUM_PREFIX "post_num_"
#define POST_SUM_PREFIX "post_sum_"
#define POST_DOT_PREFIX "post_dot_"
#define POST_MU_PREFIX "post_mu_"
#define POST_LAMBDA_PREFIX "post_lambda_"

#define CHECKPOINT_PREFIX "checkpoint_"
#define SAMPLE_PREFIX "sample_"

namespace smurff {

SaveState::SaveState(h5::File file, std::int32_t isample, bool checkpoint, bool save_aggr)
   : HDF5Group(file.createGroup(std::string(checkpoint ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(isample)))
   , m_file(file) 
   , m_isample(isample)
   , m_checkpoint(checkpoint)
   , m_save_aggr(save_aggr)
{
   m_group.createAttribute<int>(IS_CHECKPOINT_TAG, m_checkpoint);
   m_group.createAttribute<int>(NUMBER_TAG, m_isample);
}

SaveState::SaveState(h5::File file, h5::Group group)
   : HDF5Group(group), m_file(file)
{
   group.getAttribute(NUMBER_TAG).read(m_isample);
   int m_checkpoint_as_int;
   group.getAttribute(IS_CHECKPOINT_TAG).read(m_checkpoint_as_int);
   m_checkpoint = m_checkpoint_as_int;
}

SaveState::~SaveState()
{
   if (isCheckpoint())
   {
      m_file.getAttribute(LAST_CHECKPOINT_TAG).write(getName());
   }

   m_file.flush();
}

//name methods
unsigned SaveState::getNModes() const
{
   unsigned nmodes;
   m_group.getAttribute(NUM_MODES_TAG).read(nmodes);
   return nmodes;
}

bool SaveState::hasModel(std::uint64_t index) const
{
   return hasDataSet(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(index));
}

void SaveState::readModel(std::uint64_t index, Matrix &m) const
{
   read(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(index), m);
}

std::string SaveState::getName() const
{
   return std::string(isCheckpoint() ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(getIsample());
}

void SaveState::putModel(const std::vector<Matrix> &F)
{
   m_group.createAttribute(NUM_MODES_TAG, F.size());
   for (std::uint64_t m = 0; m < F.size(); ++m)
   {
      write(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(m), F[m]);
   }
}

void SaveState::readLinkMatrix(std::uint32_t mode, Matrix &m) const
{
      read(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(mode), m);
}

void SaveState::readMu(std::uint64_t index, Vector &v) const
{
   read(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index), v);
}

void SaveState::putLinkMatrix(std::uint64_t index, const Matrix &M)
{
   write(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(index), M);
}

void SaveState::putMu(std::uint64_t index, const Matrix &M) 
{
   write(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index), M);
}

bool SaveState::hasAggr(std::uint64_t index) const
{
   return hasDataSet(LATENTS_SEC_TAG, POST_NUM_PREFIX + std::to_string(index));
}

void SaveState::readAggr(std::uint64_t index, int &num, Matrix &sum, Matrix &dot) const
{
   getGroup(LATENTS_SEC_TAG).getAttribute(POST_NUM_PREFIX + std::to_string(index)).read(num);
   read(LATENTS_SEC_TAG, POST_SUM_PREFIX + std::to_string(index), sum);
   read(LATENTS_SEC_TAG, POST_DOT_PREFIX + std::to_string(index), dot);
}

void SaveState::putAggr(std::uint64_t index, int num, const Matrix &sum, const Matrix &dot)
{
   addGroup(LATENTS_SEC_TAG).createAttribute(POST_NUM_PREFIX + std::to_string(index), num);
   write(LATENTS_SEC_TAG, POST_SUM_PREFIX + std::to_string(index), sum);
   write(LATENTS_SEC_TAG, POST_DOT_PREFIX + std::to_string(index), dot);
}

void SaveState::readPostMuLambda(std::uint64_t index, Matrix &mu, Matrix &Lambda) const
{
   read(LATENTS_SEC_TAG, POST_MU_PREFIX + std::to_string(index), mu);
   read(LATENTS_SEC_TAG, POST_LAMBDA_PREFIX + std::to_string(index), Lambda);
}

void SaveState::putPostMuLambda(std::uint64_t index, const Matrix &mu, const Matrix &Lambda)
{
   write(LATENTS_SEC_TAG, POST_MU_PREFIX + std::to_string(index), mu);
   write(LATENTS_SEC_TAG, POST_LAMBDA_PREFIX + std::to_string(index), Lambda);
}


bool SaveState::hasPred() const
{
   return hasDataSet(PRED_SEC_TAG, PRED_AVG_TAG);
}

void SaveState::putPredState(double rmse_avg, double rmse_1sample, double auc_avg, double auc_1sample,
                            int sample_iter, int burnin_iter)
{
   auto pred_group = addGroup(PRED_SEC_TAG);
   pred_group.createAttribute<double>(RMSE_AVG_TAG, rmse_avg);
   pred_group.createAttribute<double>(RMSE_1SAMPLE_TAG, rmse_1sample);
   pred_group.createAttribute<double>(AUC_AVG_TAG, auc_avg);
   pred_group.createAttribute<double>(AUC_1SAMPLE_TAG, auc_1sample);
   pred_group.createAttribute<int>(SAMPLE_ITER_TAG, sample_iter);
   pred_group.createAttribute<int>(BURNIN_ITER_TAG, burnin_iter);
}

void SaveState::getPredState(
   double &rmse_avg, double &rmse_1sample, double &auc_avg, double &auc_1sample, int &sample_iter, int &burnin_iter) const
{
   auto pred_group = getGroup(PRED_SEC_TAG);
   pred_group.getAttribute(RMSE_AVG_TAG).read(rmse_avg);
   pred_group.getAttribute(RMSE_1SAMPLE_TAG).read(rmse_1sample);
   pred_group.getAttribute(AUC_AVG_TAG).read(auc_avg);
   pred_group.getAttribute(AUC_1SAMPLE_TAG).read(auc_1sample);
   pred_group.getAttribute(SAMPLE_ITER_TAG).read(sample_iter);
   pred_group.getAttribute(BURNIN_ITER_TAG).read(burnin_iter);

}

void SaveState::putPredAvgVar(const SparseMatrix &avg, const SparseMatrix &var, const SparseMatrix &one_sample)
{
   write(PRED_SEC_TAG, PRED_AVG_TAG, avg);
   write(PRED_SEC_TAG, PRED_VAR_TAG, var);
   write(PRED_SEC_TAG, PRED_ONE_TAG, one_sample);
}


void SaveState::readPredAvg(Matrix &m) const
{
   read(PRED_SEC_TAG, PRED_AVG_TAG, m);
}

void SaveState::readPredVar(Matrix &m) const
{
   read(PRED_SEC_TAG, PRED_VAR_TAG, m);
}

//getters

std::int32_t SaveState::getIsample() const
{
   return m_isample;
}

bool SaveState::isCheckpoint() const
{
   return m_checkpoint;
}

bool SaveState::saveAggr() const
{
   return m_save_aggr;
}

} // end namespace
