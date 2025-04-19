#include "core/run_parameters.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/centered_moments.hpp"

void writeTrainedModel(RunParameters params,
                       FullEnsembleTrainer model,
                       CenteredMoments m_data,
                       CenteredMoments m_model);
