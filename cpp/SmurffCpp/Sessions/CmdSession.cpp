#include <string>
#include <iostream>
#include <sstream>

#ifdef ENABLE_TESTS
#include <catch2/catch_session.hpp>
#endif

#ifdef HAVE_BOOST
#include <boost/program_options.hpp>
#endif

#include "CmdSession.h"
#include <SmurffCpp/Predict/PredictSession.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Utils/StateFile.h>
#include <SmurffCpp/Utils/StringUtils.h>

namespace smurff {

static const std::string RESTORE_NAME = "restore-from";
static const std::string BIST_NAME = "bist";

#ifdef HAVE_BOOST

static const std::string PREDICT_NAME = "predict";
static const std::string ROW_FEAT_NAME = "row-features";
static const std::string COL_FEAT_NAME = "col-features";
static const std::string HELP_NAME = "help";
static const std::string PRIOR_NAME = "prior";
static const std::string TEST_NAME = "test";
static const std::string TRAIN_NAME = "train";
static const std::string BURNIN_NAME = "burnin";
static const std::string NSAMPLES_NAME = "nsamples";
static const std::string NUM_LATENT_NAME = "num-latent";
static const std::string NUM_THREADS_NAME = "num-threads";
static const std::string SAVE_NAME = "save-name";
static const std::string SAVE_FREQ_NAME = "save-freq";
static const std::string CHECKPOINT_FREQ_NAME = "checkpoint-freq";
static const std::string THRESHOLD_NAME = "threshold";
static const std::string VERBOSE_NAME = "verbose";
static const std::string VERSION_NAME = "version";
static const std::string SEED_NAME = "seed";

namespace po = boost::program_options;

po::options_description get_desc()
{
    po::options_description general_desc("General parameters");
    general_desc.add_options()
	(VERSION_NAME.c_str(), "print version info (and exit)")
	(HELP_NAME.c_str(), "show this help information (and exit)")
	(NUM_THREADS_NAME.c_str(), po::value<int>()->default_value(Config::NUM_THREADS_DEFAULT_VALUE), "number of threads (0 = default by OpenMP)")
	(VERBOSE_NAME.c_str(), po::value<int>()->default_value(Config::VERBOSE_DEFAULT_VALUE), "verbosity of output (0, 1, 2 or 3)")
	(SEED_NAME.c_str(), po::value<int>()->default_value(Config::RANDOM_SEED_DEFAULT_VALUE), "random number generator seed");

    po::options_description train_desc("Used during training");
    train_desc.add_options()
	(TRAIN_NAME.c_str(), po::value<std::string>(), "train data file")
	(TEST_NAME.c_str(), po::value<std::string>(), "test data")
	(PRIOR_NAME.c_str(), po::value<std::vector<std::string>>()->multitoken(), "provide a prior-type for each dimension of train; prior-types:  <normal|normalone|spikeandslab|macau|macauone>")
	(BURNIN_NAME.c_str(), po::value<int>()->default_value(Config::BURNIN_DEFAULT_VALUE), "number of samples to discard")
	(NSAMPLES_NAME.c_str(), po::value<int>()->default_value(Config::NSAMPLES_DEFAULT_VALUE), "number of samples to collect")
	(NUM_LATENT_NAME.c_str(), po::value<int>()->default_value(Config::NUM_LATENT_DEFAULT_VALUE), "number of latent dimensions")
	(THRESHOLD_NAME.c_str(), po::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation");

    po::options_description predict_desc("Used during prediction");
    predict_desc.add_options()
	(PREDICT_NAME.c_str(), po::value<std::string>(), "sparse matrix with values to predict")
	(ROW_FEAT_NAME.c_str(), po::value<std::string>(), "sparse/dense row features for out-of-matrix predictions")
	(COL_FEAT_NAME.c_str(), po::value<std::string>(), "sparse/dense col features for out-of-matrix predictions")
	(THRESHOLD_NAME.c_str(), po::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation");

    po::options_description save_desc("Storing models and predictions");
    save_desc.add_options()
	(RESTORE_NAME.c_str(), po::value<std::string>(), "restore trainSession from a saved .h5 file")
	(SAVE_NAME.c_str(), po::value<std::string>()->default_value(Config::SAVE_NAME_DEFAULT_VALUE), "save model and/or predictions to this .h5 file")
	(SAVE_FREQ_NAME.c_str(), po::value<int>()->default_value(Config::SAVE_FREQ_DEFAULT_VALUE), "save every n iterations (0 == never, -1 == final model)")
	(CHECKPOINT_FREQ_NAME.c_str(), po::value<int>()->default_value(Config::CHECKPOINT_FREQ_DEFAULT_VALUE), "save state every n seconds, only one checkpointing state is kept");

    po::options_description desc("SMURFF: Scalable Matrix Factorization Framework\n\thttp://github.com/ExaScience/smurff");
    desc.add(general_desc);
    desc.add(train_desc);
    desc.add(predict_desc);
    desc.add(save_desc);


#ifdef ENABLE_TESTS
    po::options_description bist_desc("Running the built-in self test (BIST)");
    bist_desc.add_options()
	(BIST_NAME.c_str(), "run built-in self test");
    desc.add(bist_desc);
#endif // ENABLE_TESTS

    return desc;
}


struct ConfigFiller
{
    const po::variables_map &vm;
    Config &config;

    template <typename T, void (Config::*Func)(T)>
    void set(std::string name)
    {
        if (vm.count(name) && !vm[name].defaulted())
            (config.*Func)(vm[name].as<T>());
    }

    template <void (Config::*Func)(std::shared_ptr<DataConfig>)>
    void set_data(std::string name)
    {
        if (vm.count(name) && !vm[name].defaulted())
        {
            //auto tensor_config = generic_io::read_data_config(vm[name].as<std::string>(), true);
            auto tensor_config = std::make_shared<DataConfig>();
            (this->config.*Func)(tensor_config);
        }
    }
    template <DataConfig& (Config::*Func)(void)>
    void fill_data(std::string name)
    {
        if (vm.count(name) && !vm[name].defaulted())
        {
            //auto tensor_config = generic_io::read_data_config(vm[name].as<std::string>(), true);
            auto tensor_config = DataConfig();
            (this->config.*Func)() = tensor_config;
        }
    }

    void set_priors(std::string name)
    {
        if (vm.count(name) && !vm[name].defaulted())
            config.setPriorTypes(vm[name].as<std::vector<std::string>>());
    }
};

// variables_map -> Config
Config
fill_config(const po::variables_map &vm)
{
    Config config;
    ConfigFiller filler = {vm, config};


    //restore trainSession from saved state file (command line arguments are already stored in file)
    if (vm.count(RESTORE_NAME))
    {
        //restore config from state file
        std::string restore_filename = vm[RESTORE_NAME].as<std::string>();
        config.setRestoreName(restore_filename);

        //  skip if predict-trainSession
        if (!vm.count(PREDICT_NAME) && !vm.count(ROW_FEAT_NAME) && !vm.count(COL_FEAT_NAME))
            StateFile(restore_filename).restoreConfig(config);
    }

    filler.fill_data<&Config::getPredict>(PREDICT_NAME);
    filler.fill_data<&Config::getRowFeatures>(ROW_FEAT_NAME);
    filler.fill_data<&Config::getColFeatures>(COL_FEAT_NAME);
    filler.fill_data<&Config::getTest>(TEST_NAME);
    filler.fill_data<&Config::getTrain>(TRAIN_NAME);

    filler.set_priors(PRIOR_NAME);

    filler.set<double,      &Config::setThreshold>(THRESHOLD_NAME);
    filler.set<int,         &Config::setBurnin>(BURNIN_NAME);
    filler.set<int,         &Config::setNSamples>(NSAMPLES_NAME);
    filler.set<int,         &Config::setNumLatent>(NUM_LATENT_NAME);
    filler.set<int,         &Config::setNumThreads>(NUM_THREADS_NAME);
    filler.set<std::string, &Config::setRestoreName>(RESTORE_NAME);
    filler.set<std::string, &Config::setSaveName>(SAVE_NAME);
    filler.set<int,         &Config::setSaveFreq>(SAVE_FREQ_NAME);
    filler.set<int,         &Config::setCheckpointFreq>(CHECKPOINT_FREQ_NAME);
    filler.set<double,      &Config::setThreshold>(THRESHOLD_NAME);
    filler.set<int,         &Config::setVerbose>(VERBOSE_NAME);
    filler.set<int,         &Config::setRandomSeed>(SEED_NAME);

    return config;
}

// argc/argv -> variables_map -> Config
Config parse_options(int argc, char *argv[])
{
    po::variables_map vm;

    try
    {
        po::options_description desc = get_desc();

        if (argc < 2)
        {
            std::cout << desc << std::endl;
            return Config();
        }

        po::command_line_parser parser{argc, argv};
        parser.options(desc);

        po::parsed_options parsed_options = parser.run();

        store(parsed_options, vm);
        notify(vm);

        if (vm.count(HELP_NAME))
        {
            std::cout << desc << std::endl;
            return Config();
        }

        if (vm.count(VERSION_NAME))
        {
            std::cout << "SMURFF " << SMURFF_VERSION << std::endl;
            return Config();
        }
    }
    catch (const po::error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        throw(ex);
    }
    catch (std::runtime_error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        throw(ex);
    }

    const std::vector<std::string> train_only_options = {
        TRAIN_NAME, TEST_NAME, PRIOR_NAME, BURNIN_NAME, NSAMPLES_NAME, NUM_LATENT_NAME};

    //-- prediction only
    if (vm.count(PREDICT_NAME) || vm.count(COL_FEAT_NAME) || vm.count(ROW_FEAT_NAME))
    {
        if (!vm.count(RESTORE_NAME))
            THROWERROR("Need --" + RESTORE_NAME + " option in predict mode");

        for (auto name : train_only_options)
        {
            if (vm.count(name) && !vm[name].defaulted())
                THROWERROR("You're not allowed to mix train options (--" + name + ") with --" + PREDICT_NAME);
        }
    }

    return fill_config(vm);
}

#else // no BOOST

// argc/argv --> Config
Config parse_options(int argc, char *argv[])
{
    auto usage = []() {
        std::cerr << "Usage:\n"
                  << "smurff --" << RESTORE_NAME << " <saved_smurff.h5>\n"
#ifdef ENABLE_TESTS
                  << "smurff --" << BIST_NAME << " <extra Catch2 arguments>\n"
#endif // ENABLE_TESTS
                  << "(Limited smurff compiled w/o boost program options)" << std::endl;
        exit(0);
    };

    if (argc != 3) usage();

    Config config;

    //restore trainSession from saved file (config arguments are already stored in file)
    if (std::string(argv[1]) == "--" + std::string(RESTORE_NAME))
    {
        std::string RESTORE_NAME(argv[2]);
        StateFile(RESTORE_NAME).restoreConfig(config);
        config.setRestoreName(RESTORE_NAME);
     }

    else
    {
        usage();
    }

    return config;
}
#endif // NO HAVE_BOOST

// create cmd TrainSession or PredictSession
// parses args with setFromArgs, then internally creates a TrainSession or
// PredictSession from config (to validate, save, set config)
std::shared_ptr<ISession> create_cmd_session(int argc, char **argv)
{
    std::shared_ptr<ISession> session;

#ifdef ENABLE_TESTS
   if (argc >= 2 && std::string(argv[1]) == "--" + std::string(BIST_NAME)) {
        // run built-in self test
        exit(Catch::Session().run( argc - 1, argv + 1 ));
    }
#endif // ENABLE_TESTS

    Config config = parse_options(argc, argv);
    if (config.isActionTrain())
        session = std::make_shared<TrainSession>(config);
    else if (config.isActionPredict())
        session = std::make_shared<PredictSession>(config);
    else
        exit(0);

    return session;
}
} // end namespace smurff
