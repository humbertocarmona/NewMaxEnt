#include "workflows/generateSyntheticWorkFlow.hpp"
// #include "io/write_json.hpp"
#include "io/write_g_E.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/get_logger.hpp"

void generateSyntheticWorkflow(RunParameters params)
{
    auto logger = getLogger();

    // here need to make sure that the trained_model_file has:
    // h, J, q, nspins, runid
    auto data_filename = params.trained_model_file;

    // initialize core model
    MaxEntCore core(params.nspins, params.runid);

    // construct model (using FullEnsembleTrainer to generate means)
    FullEnsembleTrainer model(core, params, data_filename);
    // compute model averages in parallel
    model.computeModelAverages(1.0, true);

    model.copySyntheticMeans();
    // core.h.fill(0.0);
    // core.J.fill(0.0);
    model.saveModel("synth_");

    // auto ge = model.get_GE();
    // write_g_E(ge,params.energy_bin,"g_E.csv");

    // auto pe = model.get_PE();
    // write_g_E(pe,params.energy_bin,"p_E.csv");
}
