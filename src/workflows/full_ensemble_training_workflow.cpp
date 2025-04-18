// so now I am going to implement the full_ensemble_training_workflow:
//
// 3 - use this t setup  the FullEnsembleTrainer model
// 4 - actually train the model
// 5 - make some crude post processing
// 6 - save the result

#include "workflows/full_ensemble_training_workflow.hpp"
#include "trainers/full_ensemble_trainer.hpp"

void fullEnsembleTrainingWorkflow(RunParameters params)
{
    MaxEntCore core(params.nspins, params.runid);
    FullEnsembleTrainer model(core,
                              params.maxIterations,
                              params.tolerance_h,
                              params.tolerance_J,
                              params.eta_h,
                              params.eta_J,
                              params.alpha_h,
                              params.alpha_J,
                              params.gamma_h,
                              params.gamma_J,
                              params.raw_data_file);
}