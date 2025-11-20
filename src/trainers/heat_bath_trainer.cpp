#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"

void HeatBathTrainer::saveModel(std::string filename) const
{
    auto logger = getLogger();

    // Compute model statistics (averages)
    const_cast<HeatBathTrainer*>(this)->computeModelAverages(1.0, true);  // `const_cast` needed if this is a `const` method

    // Compute centered moments for model and data
    CenteredMoments c_model =
        computeCenteredMoments(get_m1_model(), get_m2_model(), get_m3_model());

    CenteredMoments c_data =
        computeCenteredMoments(get_m1_data(), get_m2_data(), get_m3_data());

    // Save trained model and statistics to file
    writeTrainedModel<HeatBathTrainer>(*this, c_data, c_model, filename);	

}