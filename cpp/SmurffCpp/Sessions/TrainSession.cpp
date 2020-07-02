#include "TrainSession.h"


#include <fstream>
#include <string>
#include <iomanip>

#include <SmurffCpp/Utils/omp_util.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Configs/Config.h>

#include <SmurffCpp/Priors/PriorFactory.h>

#include <SmurffCpp/result.h>
#include <SmurffCpp/StatusItem.h>

namespace smurff {

void TrainSession::init()
{
    THROWERROR_ASSERT_MSG(!m_is_init, "TrainSession::init called twice");
    
    getConfig().validate();

    if (!getConfig().getSaveName().empty())
    {
        // create state file
        m_stateFile = std::make_shared<StateFile>(getConfig().getSaveName(), true);

        //save config
        m_stateFile->saveConfig(getConfig());

        // close stateFile is we do not need it anymore
        if (!getConfig().getSaveFreq() && !getConfig().getCheckpointFreq())
            m_stateFile.reset();
    }

    // initialize pred
    if (getConfig().getTest().hasData())
    {
        m_pred = Result(getConfig().getTest());
        m_pred.setSavePred(getConfig().getSavePred());
        if (getConfig().getClassify())
            m_pred.setThreshold(getConfig().getThreshold());
    }

    // init data
    data_ptr = Data::create(getConfig().getData());
   
    // initialize priors
    std::shared_ptr<IPriorFactory> priorFactory = this->create_prior_factory();
    for (std::size_t i = 0; i < getConfig().getPriorTypes().size(); i++)
    {
        m_priors.push_back(priorFactory->create_prior(*this, i));
        m_priors.back()->setMode(i);
    }

    //init omp
    threads::init(getConfig().getVerbose(), getConfig().getNumThreads());

    //initialize random generator
    initRng();

    //init performance counters
    perf_data_init();

    //initialize test set
    m_pred.init();

    //initialize train matrix (centring and noise model)
    data().init();

    //initialize model (samples)
    model().init(getConfig().getNumLatent(), data().dim(), getConfig().getModelInitType(), getConfig().getSaveModel());

    //initialize priors
    for (auto &p : m_priors)
        p->init();

    // all basic init done
    m_is_init = true;

    //write info to console
    if (getConfig().getVerbose())
        info(std::cout, "");

    //restore trainSession (model, priors)
    bool resume = restore(m_iter);

    //print trainSession status to console
    if (getConfig().getVerbose())
        printStatus(std::cout, resume);
}

void TrainSession::run()
{
    init();
    while (step())
        ;
}

bool TrainSession::step()
{
    COUNTER("step");
    THROWERROR_ASSERT_MSG(m_is_init, "TrainSession::init() needs to be called before ::step()")

    // go to the next iteration
    m_iter++;

    bool isStep = m_iter < getConfig().getBurnin() + getConfig().getNSamples();

    if (isStep)
    {
        auto starti = tick();
        for (unsigned i=0; i<m_priors.size(); ++i)
        {
            m_priors[i]->sample_latents();
            m_priors[i]->update_prior();
        }
        
        data().update(model());
        auto endi = tick();

        //WARNING: update is an expensive operation because of sort (when calculating AUC)
        m_pred.update(m_model, m_iter < getConfig().getBurnin());

        m_secs_per_iter = endi - starti;
        m_secs_total += m_secs_per_iter;

        printStatus(std::cout);

        save();
    }

    return isStep;
}

std::ostream &TrainSession::info(std::ostream &os, std::string indent) const
{
    os << indent << name << " {\n";
    os << indent << "  Data: {" << std::endl;
    data().info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Model: {" << std::endl;
    model().info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Priors: {" << std::endl;
    for (auto &p : m_priors)
        p->info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Result: {" << std::endl;
    m_pred.info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Config: {" << std::endl;
    getConfig().info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "}\n";
    return os;
}

void TrainSession::save()
{
    //do not save if 'never save' mode is selected
    if (!getConfig().getSaveFreq() &&
        !getConfig().getCheckpointFreq())
        return;

    std::int32_t isample = m_iter - getConfig().getBurnin() + 1;
    std::int32_t niter = getConfig().getBurnin() + getConfig().getNSamples();

    //save if checkpoint threshold overdue
    if (
         getConfig().getCheckpointFreq() && 
         (
             ((tick() - m_lastCheckpointTime) >= getConfig().getCheckpointFreq()) ||
             (m_iter == niter - 1) // also save checkpoint in last iteration
         )
       )
    {
        std::int32_t icheckpoint = m_iter + 1;

        //save this iteration
        saveInternal(icheckpoint, true);

        //remove previous iteration if required (initial m_lastCheckpointIter is -1 which means that it does not exist)
        m_stateFile->removeOldCheckpoints();

        //upddate counters
        m_lastCheckpointTime = tick();
        m_lastCheckpointIter = m_iter;
    }

    //save model during sampling stage
    if (getConfig().getSaveFreq() && isample > 0)
    {
        //save_freq > 0: check modulo - do not save if not a save iteration
        if (getConfig().getSaveFreq() > 0 && (isample % getConfig().getSaveFreq()) != 0)
        {
            // don't save
        }
        //save_freq < 0: save last iter - do not save if (final model) mode is selected and not a final iteration
        else if (getConfig().getSaveFreq() < 0 && isample < getConfig().getNSamples())
        {
            // don't save
        }
        else
        {
            //do save this iteration
            saveInternal(isample, false);
        }
    }

    // close after final step
    if (m_iter == niter - 1) m_stateFile.reset();
}

void TrainSession::saveInternal(int iteration, bool checkpoint)
{
    bool final = iteration == getConfig().getNSamples();
    SaveState saveState = m_stateFile->createStep(iteration, checkpoint, final || checkpoint);

    if (getConfig().getVerbose())
    {
        std::cout << "-- Saving model, predictions,... into '" << m_stateFile->getPath() << "'." << std::endl;
    }
    double start = tick();

    m_model.save(saveState);
    m_pred.save(saveState);
    for (auto &p : m_priors) p->save(saveState);

    double stop = tick();
    if (getConfig().getVerbose())
    {
        std::cout << "-- Done saving model. Took " << stop - start << " seconds." << std::endl;
    }
}

bool TrainSession::restore(int &iteration)
{
    //if there is nothing to restore - start from initial iteration
    iteration = -1;

    //to keep track at what time we last checkpointed
    m_lastCheckpointTime = tick();
    m_lastCheckpointIter = -1;

    if (getConfig().getRestoreName().empty())
        return false;

    // open state file
    StateFile restoreFile(getConfig().getRestoreName());

    if (!restoreFile.hasCheckpoint())
        return false;

    SaveState saveState = restoreFile.openCheckpoint();
    if (getConfig().getVerbose())
        std::cout << "-- Restoring model, predictions,... from '" << restoreFile.getPath() << "'." << std::endl;

    m_model.restore(saveState);
    m_pred.restore(saveState);
    for (auto &p : m_priors)
        p->restore(saveState);

    //restore last iteration index
    if (saveState.isCheckpoint())
    {
        iteration = saveState.getIsample() - 1; //restore original state

        //to keep track at what time we last checkpointed
        m_lastCheckpointTime = tick();
        m_lastCheckpointIter = iteration;
    }
    else
    {
        iteration = saveState.getIsample() + getConfig().getBurnin() - 1; //restore original state

        //to keep track at what time we last checkpointed
        m_lastCheckpointTime = tick();
        m_lastCheckpointIter = iteration;
    }

    return true;
}

const Result &TrainSession::getResult() const
{
   return m_pred;
}

StatusItem TrainSession::getStatus() const
{
    THROWERROR_ASSERT(m_is_init);

    StatusItem ret;

    if (m_iter < 0)
    {
        ret.phase = "Initial";
        ret.iter = m_iter + 1;
        ret.phase_iter = 0;
    }
    else if (m_iter < getConfig().getBurnin())
    {
        ret.phase = "Burnin";
        ret.iter = m_iter + 1;
        ret.phase_iter = getConfig().getBurnin();
    }
    else
    {
        ret.phase = "Sample";
        ret.iter = m_iter - getConfig().getBurnin() + 1;
        ret.phase_iter = getConfig().getNSamples();
    }

    for (int i = 0; i < (int)model().nmodes(); ++i)
    {
        ret.model_norms.push_back(model().U(i).norm());
    }

    ret.train_rmse = data().train_rmse(model());

    ret.rmse_avg = m_pred.rmse_avg;
    ret.rmse_1sample = m_pred.rmse_1sample;

    ret.auc_avg = m_pred.auc_avg;
    ret.auc_1sample = m_pred.auc_1sample;

    ret.elapsed_iter = m_secs_per_iter;
    ret.elapsed_total = m_secs_total;

    ret.nnz_per_sec = (double)(data().nnz()) / m_secs_per_iter;
    ret.samples_per_sec = (double)(model().nsamples()) / m_secs_per_iter;

    return ret;
}

void TrainSession::printStatus(std::ostream &output, bool resume)
{
    if (!getConfig().getVerbose())
        return;

    auto status_item = getStatus();

    std::string resumeString = resume ? "Continue from " : std::string();

    if (getConfig().getVerbose() > 0)
    {
        if (m_iter < 0)
        {
            output << " ====== Initial phase ====== " << std::endl;
        }
        else if (m_iter < getConfig().getBurnin() && m_iter == 0)
        {
            output << " ====== Sampling (burning phase) ====== " << std::endl;
        }
        else if (m_iter == getConfig().getBurnin())
        {
            output << " ====== Burn-in complete, averaging samples ====== " << std::endl;
        }

        output << resumeString << status_item.asString() << std::endl;

        if (getConfig().getVerbose() > 1)
        {
            output << std::fixed << std::setprecision(4) << "  RMSE train: " << status_item.train_rmse << std::endl;
            output << "  Priors:" << std::endl;

            for (const auto &p : m_priors)
                p->status(output, "     ");

            output << "  Model:" << std::endl;
            model().status(output, "    ");
            output << "  Noise:" << std::endl;
            data().status(output, "    ");
        }

        if (getConfig().getVerbose() > 2)
        {
            output << "  Compute Performance: " << status_item.samples_per_sec << " samples/sec, " << status_item.nnz_per_sec << " nnz/sec" << std::endl;
        }
    }
}

std::string StatusItem::getCsvHeader()
{
    return "phase;iter;phase_len;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;elapsed_1samp;elapsed_total";
}

std::string StatusItem::asCsvString() const
{
    char ret[1024];
    snprintf(ret, 1024, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%0.1f;%0.1f",
             phase.c_str(), iter, phase_iter, rmse_avg, rmse_1sample, train_rmse,
             auc_1sample, auc_avg, elapsed_iter, elapsed_total);

    return ret;
}

void TrainSession::initRng()
{
    //init random generator
    if (getConfig().getRandomSeedSet())
        init_bmrng(getConfig().getRandomSeed());
    else
        init_bmrng();
}

std::shared_ptr<IPriorFactory> TrainSession::create_prior_factory() const
{
    return std::make_shared<PriorFactory>();
}
} // end namespace smurff
