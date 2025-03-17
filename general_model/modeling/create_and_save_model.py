# """
# Train, validate and tune hyperparameters of the model to be created on the
# defined respective dataset partitions, eventually saving a new final model
# trained on both the training and the validation set with the (best, final)
# hyperparameters previously found to optimize the metric on the validation set
# and reporting its performances on the previously unseen test set - the
# executed instructions produce purely deterministic results as all sources of
# stochasticity have their seeds fixed.
# """


# # pylint: disable=import-error,no-name-in-module
# from tensorflow.config.optimizer import set_jit
# # pylint: enable=import-error,no-name-in-module

# from constants import (
#     N_EPOCHS_TO_REACH_IN_TOTAL,
#     N_HYPERPARAMETER_COMBINATIONS_SAMPLED,
#     MODEL_FILE_PATH,
#     fix_seeds_for_reproducible_results
# )
# from hyperparameter_tuning import (
#     hyperparametrized_training_and_validation,
#     tune_hyperparameters
# )


# def main() -> None:
#     """
#     Train, validate and tune hyperparameters of the model to be created on the
#     defined respective dataset partitions, eventually saving a new final model
#     trained on both the training and the validation set with the (best, final)
#     hyperparameters previously found to optimize the metric on the validation
#     set and reporting its performances on the previously unseen test set - the
#     executed instructions produce purely deterministic results as all sources
#     of stochasticity have their seeds fixed.
#     """
#     fix_seeds_for_reproducible_results()

#     # optimizing performances with XLA for any possible graph (automatically
#     # identifying and compiling clusters of nodes):
#     set_jit('autoclustering')

#     best_hyperparameters = tune_hyperparameters(
#         max_n_epochs_per_trial=N_EPOCHS_TO_REACH_IN_TOTAL,
#         n_trials=N_HYPERPARAMETER_COMBINATIONS_SAMPLED
#     )

#     final_model = hyperparametrized_training_and_validation(
#         config=best_hyperparameters,
#         train_also_on_validation_data_and_validate_on_test_data=True
#     )

#     final_model.save(
#         filepath=MODEL_FILE_PATH,
#         overwrite=False
#     )


# if __name__ == '__main__':
#     main()
