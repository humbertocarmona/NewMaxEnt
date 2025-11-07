#include "io/write_json.hpp"
#include "trainers/heat_bath_pretrain.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "trainers/wang_landau_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/training_workflow.hpp"
#include <fstream>

void WangLandauTrainingWorkflow(RunParameters params)
{
    auto logger        = getLogger();
    auto data_filename = params.raw_data_file;
    if (data_filename == "none")
        data_filename = params.trained_model_file;

    MaxEntCore core(params.nspins, params.runid);
    pretrain_with_heatbath(core, params, data_filename);

    WangLandauTrainer model(core, params, data_filename);

    // model.computeDensityOfStates();

    // auto log_g_E = model.get_log_g_E();

    // std::map<int, double> sorted_log_g_E(log_g_E.begin(), log_g_E.end());

    // std::ofstream out("log_g_E.csv");

    // out << "E,G\n";
    // for (const auto &pair : sorted_log_g_E)
    // {
    //     out << pair.first*params.energy_bin << "," << pair.second << "\n";
    // }
    // out.close();

    model.train();
}
