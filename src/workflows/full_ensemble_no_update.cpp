#include "workflows/full_ensemble_no_update.hpp"
#include "io/make_file_names.hpp"
#include "io/write_g_E.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/get_logger.hpp"

// --------------------------------------------------------
// HERE parameters are not updated!!!
//      h = h_read_from_file
//      J = J_read_from_file
//
//   For run_type ==  Gen_Full or Gen_MC, the input file has
//   only nspins,  h and J ( the synthetic values).
//   In this case m1_data, m2_data, m3_data and pK_data are
//   set to Zero by default. then data -< model in the end
//
// --------------------------------------------------------
void full_ensemble_no_update(RunParameters params)
{
    auto logger = getLogger();
    auto gen    = (params.run_type == "Gen_Full" || params.run_type == "Gen_MC");
    // here need to make sure that the trained_model_file has:
    // h, J, q, nspins, runid
    auto data_filename = params.trained_model_file;

    // initialize core model
    MaxEntCore core(params.nspins, params.runid);


    // construct model (using FullEnsembleTrainer to generate means)
    FullEnsembleTrainer model(core, params, data_filename);

    auto m1_orig = model.get_m1_model();
    auto m2_orig = model.get_m2_model();
    auto m3_orig = model.get_m3_model();
    auto pk_orig = model.get_pK_model();
    // compute model averages in parallel
    logger->info("computing model averages");
    model.computeModelAverages(1.0, true);
    logger->info("... finished");


    //* in this case original data = {0}
    if (gen)
        model.computed_means_to_data();

    //* Copy will keep the originally computed means...
    if (params.run_type == "Copy")
        model.set_model_means(m1_orig, m2_orig, m3_orig, pk_orig);

    logger->info("before save model");
    model.saveModel(params.file_final, false);
}
