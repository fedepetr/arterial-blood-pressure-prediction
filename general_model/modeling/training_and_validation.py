"""
Training of a model instance of the defined architecture on the specified
training set and saving of the trained model to the file system.
"""


from json import dump as json_dump
from math import ceil, exp
from os import listdir, makedirs
from os.path import exists as path_exists, join as path_join
from shutil import copy2
from sys import exit as sys_exit
from typing import Any, Dict, Literal, Type, get_args

# pylint: disable=no-name-in-module,import-error
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
# pylint: enable=no-name-in-module,import-error

from constants import (
    ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME,
    ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM,
    CUSTOM_OBJECTS_IN_MODEL_DEFINITION,
    DOUBLE_UNET_VARIANT,
    DOUBLE_UNET_VARIANT_SUBMODEL_NAMES,
    HYPERPARAMETERS_FILE_PATH,
    INPUTS_STANDARDIZATION_LAYER_NAME,
    LEARNING_RATE_INITIAL_VALUE,
    LEARNING_RATE_FINAL_VALUE,
    LEARNING_RATE_TREND_ENABLED,
    MINI_BATCH_SIZE,
    N_EPOCHS_TO_REACH_IN_TOTAL,
    OPTIMIZER_ID,
    OPTIMIZER_ID_FOR_ADAM,
    OPTIMIZER_PARAMETERS,
    OUTPUT_NAME_FOR_DBP,
    OUTPUT_NAME_FOR_SBP,
    UCI_TRAINING_SET_DIR,
    ValidArgumentsForDataSource,
    ValidArgumentsForInputDomain,
    ValidArgumentsForOptimizerId,
    ValidArgumentsForTask,
    fix_seeds_for_reproducible_results
)
from metrics import mean_signed_error
from model_architectures import (
    EXPERIMENT_DIR,
    MODEL_FILE_PATH,
    build_model_architecture
)
from samples_and_labels import (
    DATA_SOURCE,
    INPUT_DOMAIN,
    TASK,
    create_datasets
)


ValidArgumentsForLogFor = Literal['tuning', 'testing']


TRAINING_AND_VALIDATION_STATISTICS_DIR = path_join(
    EXPERIMENT_DIR,
    'training_and_validation_statistics'
)
TRAINING_AND_VALIDATION_STATISTICS_SUMMARY_PATH = path_join(
    EXPERIMENT_DIR,
    'training_and_validation_statistics_summary.json'
)


def learning_rate_as_function_of_epoch_index_wrapper(
        initial_learning_rate: float,
        final_plateau_learning_rate: float
):
    """
    Wrapper to get the learning rate trend as function of epoch index with
    fixed initial and final learning rate values.
    """
    def learning_rate_as_function_of_epoch_index(
            epoch_index: int,
            *args
    ) -> float:
        """
        Return the learning rate value corresponding to the input epoch index,
        according to the herein defined learning rate trend with epochs.
        """
        return (
            final_plateau_learning_rate + (
                (initial_learning_rate - final_plateau_learning_rate) /
                (1 + exp((epoch_index - 10) / 2))
            )
        )
    
    return learning_rate_as_function_of_epoch_index


def train_and_validate_model(  # pylint: disable=too-many-arguments
        model_instance: Type[Model],
        training_set: Dataset,
        validation_set: Dataset,
        data_source: ValidArgumentsForDataSource,
        input_domain: ValidArgumentsForInputDomain,
        task: ValidArgumentsForTask,
        learning_rate_trend_enabled: bool,
        learning_rate_initial_value: float,
        learning_rate_final_value: float,
        learning_rate_static_value: float,
        n_epochs: int,
        optimizer_id: ValidArgumentsForOptimizerId,
        optimizer_kwargs: Dict[str, Any],
        starting_epoch_index: int,
        log_for: ValidArgumentsForLogFor
) -> Type[Model]:
    """
    Train the input model instance on the input training set, eventually
    saving the trained model to the file system.
    """
    assert (
        input_domain in get_args(tp=ValidArgumentsForInputDomain)
    ), "Invalid 'input_domain' argument passed."
    assert (
        log_for in get_args(tp=ValidArgumentsForLogFor)
    ), "Invalid 'log_for' argument passed."
    assert (
        task in get_args(tp=ValidArgumentsForTask)
    ), "Invalid 'task' argument passed."

    metric_name = 'mean_absolute_error'
    metric_name_on_validation_set = 'val_' + metric_name

    # defining loss function, optimizer and metrics, while JIT-compiling
    # the model for a computationally-optimized training:
    adam_over_sgd = optimizer_id == OPTIMIZER_ID_FOR_ADAM
    optimizer_kwargs['learning_rate'] = (
        learning_rate_initial_value if learning_rate_trend_enabled
        else learning_rate_static_value
    )
    compilation_kwargs = {
        'loss': (
            MeanSquaredError() if task == ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM
            else {
                OUTPUT_NAME_FOR_DBP: MeanSquaredError(),
                OUTPUT_NAME_FOR_SBP: MeanSquaredError()
            }  # equally weighted (no 'loss_weights' argument to '.compile()')
        ),
        'optimizer': (
            # only if the training starts from scratch, i.e. it is not
            # resumed from a previous state:
            (
                Adam(**optimizer_kwargs)
                if adam_over_sgd else
                SGD(**optimizer_kwargs)
            ) if starting_epoch_index == 0 else model_instance.optimizer
        ),
        'jit_compile': True
    }
    metrics = [
        MeanAbsoluteError(name=metric_name),
        mean_signed_error
    ]
    if data_source == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE:
        compilation_kwargs['weighted_metrics'] = metrics
    else:
        compilation_kwargs['metrics'] = metrics
    model_instance.compile(**compilation_kwargs)

    if starting_epoch_index == 0:
        # if considering the time-domain input representation:
        if input_domain == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME:
            # computing the initial normalization layer's statistics (the
            # standardization mean and variance of each input feature) over
            # the training set, so as to have them both at training and at
            # inference time:
            if DOUBLE_UNET_VARIANT:
                model_instance.get_layer(
                    name=DOUBLE_UNET_VARIANT_SUBMODEL_NAMES[0]
                ).get_layer(
                    name=INPUTS_STANDARDIZATION_LAYER_NAME
                ).adapt(
                    data=training_set.map(
                        # selecting only samples:
                        lambda *mini_batch_data: mini_batch_data[0]
                    )
                )
            else:
                model_instance.get_layer(
                    name=INPUTS_STANDARDIZATION_LAYER_NAME
                ).adapt(
                    data=training_set.map(
                        # selecting only samples:
                        lambda *mini_batch_data: mini_batch_data[0]
                    )
                )

    training_callbacks = [
        # callback that logs the training and validation statistics of Keras
        # at the end of each training epoch for TensorBoard:
        TensorBoard(
            log_dir=TRAINING_AND_VALIDATION_STATISTICS_DIR
        ),
        # callback to save (or override) the best model obtained so far:
        ModelCheckpoint(
            filepath=MODEL_FILE_PATH,
            monitor='val_' + metric_name,
            save_best_only= True,
            save_weights_only=False,
            mode='min',
            save_freq='epoch',
            initial_value_threshold=20  # [mmHg]
        ),
    ]
    if learning_rate_trend_enabled:
        training_callbacks.append(
            # callback for a learning rate trend that varies with epochs:
            LearningRateScheduler(
                schedule=learning_rate_as_function_of_epoch_index_wrapper(
                    initial_learning_rate=learning_rate_initial_value,
                    final_plateau_learning_rate=learning_rate_final_value
                )
            )
        )

    # training the model while validating it:
    training_history = model_instance.fit(
        x=training_set,
        epochs=n_epochs,
        initial_epoch=starting_epoch_index,
        callbacks=training_callbacks,
        validation_data=validation_set,
        verbose=1
    )

    with open(TRAINING_AND_VALIDATION_STATISTICS_SUMMARY_PATH, 'w') as file:
        json_dump(obj=training_history.history, fp=file)

    return model_instance


if __name__ == '__main__':
    fix_seeds_for_reproducible_results()

    # ------------------------------------------------------------------------
    # learning rate plot
    # ------------------------------------------------------------------------
    # from sys import argv, exit

    # if len(argv) > 1 and argv[1] == 'lr':
    #     from matplotlib.pyplot import figure, plot, show

    #     # plotting the learning rate trend with epochs:
    #     learning_rate_as_function_of_epoch_index = (
    #         learning_rate_as_function_of_epoch_index_wrapper(   
    #             initial_learning_rate=LEARNING_RATE_INITIAL_VALUE,
    #             final_plateau_learning_rate=LEARNING_RATE_FINAL_VALUE,
    #         )
    #     )
    #     epoch_indexes = list(range(20))
    #     learning_rate_values = [
    #         learning_rate_as_function_of_epoch_index(epoch_index) for
    #         epoch_index in epoch_indexes
    #     ]
    #     figure()
    #     plot(epoch_indexes, learning_rate_values)
    #     show()

    #     # terminating the program:
    #     exit("Done!")
    # ------------------------------------------------------------------------

    training_dataset, validation_dataset, test_dataset = create_datasets(
        mini_batch_size=MINI_BATCH_SIZE
    )

    # in case an experiment directory with the same name (thus path) of the
    # experiment to be executed already exists:
    if path_exists(path=EXPERIMENT_DIR):
        sys_exit(
            "❌⚠️ An experiment with the same name already exists! Make " +
            "sure to use unique experiment identifiers. Resuming trainings " +
            "of past experiments is not supported at the moment. ⚠️❌"
        )

        if DATA_SOURCE == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE:
            raise NotImplementedError("Yet to be sorted out.")
        model = load_model(
            filepath=MODEL_FILE_PATH,
            custom_objects=CUSTOM_OBJECTS_IN_MODEL_DEFINITION,
            compile=True
        )

        n_training_iterations_done_so_far = (
            model.optimizer.iterations.numpy().item()
        )
        n_training_mini_batches_per_epoch = ceil(
            # number of training samples (there's a .gitignore too):
            (len(listdir(UCI_TRAINING_SET_DIR)) - 1) /
            # number of samples per full mini-batch:
            MINI_BATCH_SIZE
        )
        next_epoch_index, check = divmod(
            n_training_iterations_done_so_far,
            n_training_mini_batches_per_epoch
        )
        assert check == 0, "Last epoch(s) not carried out entirely."
        print(
            '\n\n',
            f"{next_epoch_index} epoch done previously",
            "->",
            f"starting epoch idex: {next_epoch_index}",
            end='\n\n'
        )

    else:
        makedirs(name=EXPERIMENT_DIR)
        makedirs(name=TRAINING_AND_VALIDATION_STATISTICS_DIR)

        copy2(src=HYPERPARAMETERS_FILE_PATH, dst=EXPERIMENT_DIR)

        model = build_model_architecture(
            domain=INPUT_DOMAIN,
            task=TASK,
            data=DATA_SOURCE
        )

        next_epoch_index = 0

    model = train_and_validate_model(
        model_instance=model,
        training_set=training_dataset,
        validation_set=validation_dataset,
        data_source=DATA_SOURCE,
        input_domain=INPUT_DOMAIN,
        task=TASK,
        learning_rate_trend_enabled=LEARNING_RATE_TREND_ENABLED,
        learning_rate_initial_value=LEARNING_RATE_INITIAL_VALUE,
        learning_rate_final_value=LEARNING_RATE_FINAL_VALUE,
        learning_rate_static_value=LEARNING_RATE_FINAL_VALUE,
        n_epochs=N_EPOCHS_TO_REACH_IN_TOTAL,
        optimizer_id=OPTIMIZER_ID,
        optimizer_kwargs=OPTIMIZER_PARAMETERS,
        starting_epoch_index=next_epoch_index,
        log_for='testing'
    )
