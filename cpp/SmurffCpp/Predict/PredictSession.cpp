#include <memory>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/StateFile.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/ResultItem.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff
{

PredictSession::PredictSession(const std::string &model_file)
    : ISession(Config()) //FIXME
    , m_model_file(StateFile(model_file))
    , m_pred_savefile()
    , m_has_config(false)
    , m_num_latent(-1)
    , m_dims(PVec<>(0))
{
    m_stepfiles = m_model_file.openSampleSteps();
}

PredictSession::PredictSession(const Config &config)
    : ISession(config)
    , m_model_file(StateFile(config.getRestoreName()))
    , m_pred_savefile(std::make_unique<StateFile>(config.getSaveName()))
    , m_has_config(true)
    , m_num_latent(-1)
    , m_dims(PVec<>(0))
{
    m_stepfiles = m_model_file.openSampleSteps();
}

void PredictSession::run()
{
    THROWERROR_ASSERT(m_has_config);


    if (getConfig().getTest().hasData())
    {
        init();
        while (step())
            ;

        return;
    }
    else
    {
        std::pair<int, const DataConfig &> side_info =
            (getConfig().getRowFeatures().hasData()) ?
            std::make_pair(0, getConfig().getRowFeatures()) :
            std::make_pair(1, getConfig().getColFeatures()) ;

        THROWERROR_ASSERT_MSG(!side_info.second.hasData(), "Need either test, row features or col features");

        if (side_info.second.isDense())
        {
            const auto &dense_matrix = side_info.second.getDenseMatrixData();
            predict(side_info.first, dense_matrix, getConfig().getSaveFreq());
        }
        else
        {
            const auto &sparse_matrix = side_info.second.getSparseMatrixData();
            predict(side_info.first, sparse_matrix, getConfig().getSaveFreq());
        }
    }
}

void PredictSession::init()
{
    THROWERROR_ASSERT(m_has_config);
    THROWERROR_ASSERT(getConfig().getTest().hasData());
    m_result = Result(getConfig().getTest(), getConfig().getNSamples());

    m_pos = m_stepfiles.rbegin();
    m_iter = 0;
    m_is_init = true;

    THROWERROR_ASSERT_MSG(getConfig().getSaveName() != m_model_file.getPath(),
                          "Cannot have same output file for model and predictions - both have " + getConfig().getSaveName());

    if (getConfig().getSaveFreq())
    {
        // create save file
        m_pred_savefile = std::make_unique<StateFile>(getConfig().getSaveName(), true);
    }

    if (getConfig().getVerbose())
        info(std::cout, "");
}

bool PredictSession::step()
{
    THROWERROR_ASSERT(m_has_config);
    THROWERROR_ASSERT(m_is_init);
    THROWERROR_ASSERT(m_pos != m_stepfiles.rend());

    double start = tick();
    Model model;
    restoreModel(model, *m_pos);
    m_result.update(model, false);
    double stop = tick();
    m_iter++;
    m_secs_per_iter = stop - start;
    m_secs_total += m_secs_per_iter;

    if (getConfig().getVerbose())
        std::cout << getStatus().asString() << std::endl;

    if (getConfig().getSaveFreq() > 0 && (m_iter % getConfig().getSaveFreq()) == 0)
        save();

    auto next_pos = m_pos;
    next_pos++;
    bool last_iter = next_pos == m_stepfiles.rend();

    //save last iter
    if (last_iter && getConfig().getSaveFreq() == -1)
        save();

    m_pos++;
    return !last_iter;
}

void PredictSession::save()
{
    //save this iteration
    SaveState saveState = m_pred_savefile->createSampleStep(m_iter, false);

    if (getConfig().getVerbose())
    {
        std::cout << "-- Saving predictions into '" << m_pred_savefile->getPath() << "'." << std::endl;
    }

    m_result.save(saveState);
}

StatusItem PredictSession::getStatus() const
{
    StatusItem ret;
    ret.phase = "Predict";
    ret.iter = m_pos->getIsample();
    ret.phase_iter = m_stepfiles.size();

    ret.train_rmse = NAN;

    ret.rmse_avg = m_result.rmse_avg;
    ret.rmse_1sample = m_result.rmse_1sample;

    ret.auc_avg = m_result.auc_avg;
    ret.auc_1sample = m_result.auc_1sample;

    ret.elapsed_iter = m_secs_per_iter;
    ret.elapsed_total = m_secs_total;

    return ret;
}

const Result &PredictSession::getResult() const
{
    return m_result;
}

std::ostream &PredictSession::info(std::ostream &os, std::string indent) const
{
    os << indent << "PredictSession {\n";
    os << indent << "  Model {\n";
    os << indent << "    model-file: " << m_model_file.getPath() << "\n";
    os << indent << "    num-samples: " << getNumSteps() << "\n";
    os << indent << "    num-latent: " << getNumLatent() << "\n";
    os << indent << "    dimensions: " << getModelDims() << "\n";
    os << indent << "  }\n";
    os << indent << "  Predictions {\n";
    m_result.info(os, indent + "    ");
    if (getConfig().getSaveFreq() > 0)
    {
        os << indent << "    Save predictions: every " << getConfig().getSaveFreq() << " iteration\n";
        os << indent << "    Output file: " << getConfig().getSaveName() << "\n";
    }
    else if (getConfig().getSaveFreq() < 0)
    {
        os << indent << "    Save predictions after last iteration\n";
        os << indent << "    Output file: " << getConfig().getSaveName() << "\n";
    }
    else
    {
        os << indent << "    Don't save predictions\n";
    }
    os << indent << "  }" << std::endl;
    os << indent << "}\n";
    return os;
}

void PredictSession::restoreModel(Model &model, const SaveState &sf, int skip_mode)
{
    model.restore(sf, skip_mode);

    if (m_num_latent <= 0)
    {
        m_num_latent = model.nlatent();
        m_dims = model.getDims();
    }
    else
    {
        THROWERROR_ASSERT(m_num_latent == model.nlatent());
        THROWERROR_ASSERT(m_dims == model.getDims());
    }

    THROWERROR_ASSERT(m_num_latent > 0);
}

void PredictSession::restoreModel(Model &model, int i, int skip_mode)
{
    restoreModel(model, m_stepfiles.at(i), skip_mode);
}

// predict one element
ResultItem PredictSession::predict(PVec<> pos, const SaveState &sf)
{
    ResultItem ret{pos};
    predict(ret, sf);
    return ret;
}

// predict one element
void PredictSession::predict(ResultItem &res, const SaveState &sf)
{
    Model model;
    model.restore(sf);
    auto pred = model.predict(res.coords);
    res.update(pred);
}

// predict one element
void PredictSession::predict(ResultItem &res)
{
    auto stepfiles = m_model_file.openSampleSteps();

    for (const auto &sf : stepfiles)
        predict(res, sf);
}

ResultItem PredictSession::predict(PVec<> pos)
{
    ResultItem ret{pos};
    predict(ret);
    return ret;
}

// predict all elements in Ytest
std::shared_ptr<Result> PredictSession::predict(const DataConfig &Y)
{
    auto res = std::make_shared<Result>(Y, getNumSteps());

    for (const auto &s : m_stepfiles)
    {
        Model model;
        restoreModel(model, s);
        res->update(model, false);
    }

    return res;
}

} // end namespace smurff
