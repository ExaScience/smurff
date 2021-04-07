#pragma once

#include <memory>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Utils/StateFile.h>
#include <SmurffCpp/Sessions/ISession.h>
#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>


namespace smurff {

class Result;
struct ResultItem;

class PredictSession : public ISession
{
    friend SparseMatrix predict(const SparseMatrix &coords, const Matrix &U, const Matrix &V);

private:
    StateFile m_model_file;
    std::unique_ptr<StateFile> m_pred_savefile;
    bool m_has_config;

    Result m_result;
    std::vector<SaveState>::reverse_iterator m_pos;

    double m_secs_per_iter;
    double m_secs_total;
    int m_iter;

    std::vector<SaveState> m_stepfiles;

    int m_num_latent;
    PVec<> m_dims;

public:
    void restoreModel(Model &, const SaveState &, int skip_mode = -1);
    void restoreModel(Model &, int i, int skip_mode = -1);

public:
    int    getNumSteps()  const { return m_stepfiles.size(); } 
    int    getNumLatent() const { return m_num_latent; } 
    PVec<> getModelDims() const { return m_dims; } 

public:
    // ISession interface 
    void run() override;
    bool step() override;
    void init() override;

    StatusItem getStatus() const override;
    const Result &getResult() const override;

private:
    void save();

  public:
    PredictSession(const std::string &model_file);
    PredictSession(const Config &config);

    std::ostream& info(std::ostream &os, std::string indent) const override;

    // predict one element - based on position only
    ResultItem predict(PVec<> Ytest);
    
    ResultItem predict(PVec<> Ytest, const SaveState &sf);

    // predict one element - based on ResultItem
    void predict(ResultItem &);
    void predict(ResultItem &, const SaveState &sf);

    // predict all elements in Ytest
    std::shared_ptr<Result> predict(const DataConfig &Y);
    void predict(Result &, const SaveState &);

    // predict element or elements based on sideinfo
    template <class Feat>
    std::shared_ptr<Matrix> predict(int mode, const Feat &f, int save_freq = 0);
};

// predict element or elements based on sideinfo
template <class Feat>
std::shared_ptr<Matrix> PredictSession::predict(int mode, const Feat &f, int save_freq)
{
    std::shared_ptr<Matrix> average(nullptr);

    for (int step = 0; step < getNumSteps(); step++)
    {
        if (getConfig().getVerbose())
        {
            std::cout << "Out-of-matrix prediction step " << step << "/" << getNumSteps() << "." << std::endl;
        }
 
        Model model;
        restoreModel(model, step, mode);
        auto predictions = model.predict(mode, f);
        if (!average)
            average = std::make_shared<Matrix>(predictions);
        else
            *average += predictions;

        if (save_freq > 0 && (step % save_freq) == 0)
        {
            auto filename = getConfig().getSaveName();
            if (getConfig().getVerbose())
            {
                std::cout << "-- Saving sample " << step << " to " << filename << "." << std::endl;
            }
            //matrix_io::eigen::write_matrix(filename, predictions);
            // FIXME 
        }
    }

    (*average) /= (double)getNumSteps();

    if (save_freq != 0)
    {
        auto filename = getConfig().getSaveName();
        if (getConfig().getVerbose())
        {
            std::cout << "-- Saving average predictions to " << filename << "." << std::endl;
        }
        //matrix_io::eigen::write_matrix(filename, *average);
    }

    return average;
}

SparseMatrix predict_matrix(const SparseMatrix &, const std::vector<Matrix> &);
SparseTensor predict_tensor(const SparseTensor &, const std::vector<Matrix> &);
    

} // end namespace smurff