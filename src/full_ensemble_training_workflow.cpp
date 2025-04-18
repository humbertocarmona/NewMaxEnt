// so now I am going to implement the full_ensemble_training_workflow:
// 
// 1 - read JSON file with run parameters: nspins, location of row data
// 2 - setup core model
// 3 - use this t setup  the FullEnsembleTrainer model
// 4 - actually train the model
// 5 - make some crude post processing
// 6 - save the result