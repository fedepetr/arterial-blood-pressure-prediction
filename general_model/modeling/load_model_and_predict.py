"""
Script to load a pretrained model and employ it to run interactive predictions
via CLI.
"""


from os import listdir
from os.path import join as path_join
from random import choice, seed

from matplotlib import rcParams
from matplotlib.pyplot import subplots, show
from numpy import array, fromfile, save, squeeze, stack
# pylint: disable=no-name-in-module,import-error
from tensorflow import convert_to_tensor
from tensorflow.keras.metrics import mean_absolute_error, MeanAbsoluteError
from tensorflow.keras.models import load_model  # noqa: E402,E501
# pylint: enable=no-name-in-module,import-error

from constants import (  # noqa: E402,E501
    ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME,
    ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM,
    CUSTOM_OBJECTS_IN_MODEL_DEFINITION,
    SIGNALS_SAMPLING_FREQUENCY,
    UCI_TEST_SET_DIR,
    UCI_TRAINING_SET_DIR,
    UCI_VALIDATION_SET_DIR
)
from model_architectures import MODEL_FILE_PATH
from samples_and_labels import (
    DATA_SOURCE,
    INPUT_DOMAIN,
    TASK,
    create_datasets
)


UCI_TEST_SET_SIGNAL_BLOCK_SEGMENTS_FILENAMES = listdir(
    path=UCI_TEST_SET_DIR
)
UCI_TRAINING_SET_SIGNAL_BLOCK_SEGMENTS_FILENAMES = listdir(
    path=UCI_TRAINING_SET_DIR
)
UCI_VALIDATION_SET_SIGNAL_BLOCK_SEGMENTS_FILENAMES = listdir(
    path=UCI_VALIDATION_SET_DIR
)


if __name__ == '__main__':
    seed(a=0)
    rcParams.update({'font.size': 18})

    # loading and JIT-compiling the pretrained model:
    model = load_model(
        filepath=MODEL_FILE_PATH,  # [:-3],
        custom_objects=CUSTOM_OBJECTS_IN_MODEL_DEFINITION,
        compile=False
    )
    model.compile(metrics=[MeanAbsoluteError()], jit_compile=True)

    if DATA_SOURCE == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE:
        if (
                INPUT_DOMAIN != ARGUMENT_NAME_FOR_DOMAIN_AS_TIME or
                TASK != ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM
        ):
            raise NotImplementedError(
                f"Only '{ARGUMENT_NAME_FOR_DOMAIN_AS_TIME}' input domain " +
                f"and '{ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM}' prediction " +
                "task are supported as arguments with " +
                f"'{ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE}' data source."
            )

        # letting the user choose which set to inspect beforehand, validating
        # the choice and setting variables for filenames and folders
        # accordingly:
        chosen_set = input(
            "Which dataset do you want to inspect? [tr/val/test] "
        )
        if chosen_set == 'tr':
            chosen_set_index = 0
        elif chosen_set == 'val':
            chosen_set_index = 1
        elif chosen_set == 'test':
            chosen_set_index = 2
        else:
            raise Exception("Unexpected choice.")

        training_validation_test_sets = create_datasets(mini_batch_size=1)
        chosen_set = training_validation_test_sets[chosen_set_index]

        for mini_batch_data in chosen_set:
            for mini_batch_data in chosen_set:
                mini_batch_samples = mini_batch_data[0]
                mini_batch_labels = mini_batch_data[1]
                mini_batch_mask_as_weights = mini_batch_data[2]

                # estimating the target signal from the input signals via
                # model prediction:
                predicted_signal = model.predict_on_batch(
                    x=mini_batch_samples
                ).squeeze()

                # comparing the predicted and the target signals by visually
                # plotting them, together with the input signals:
                figure, axes = subplots(2, sharex=True)
                figure.suptitle(t=f'...')
                x_axis_values = (
                    array(list(range(mini_batch_labels.shape[1]))) /
                    SIGNALS_SAMPLING_FREQUENCY
                )
                axes[0].plot(
                    x_axis_values,
                    mini_batch_labels[0],
                    label='Target ABP'
                )
                axes[0].plot(
                    x_axis_values,
                    predicted_signal,
                    label='Predicted ABP'
                )
                axes[0].legend()
                axes[0].title.set_text(s="ABP Error")
                axes[0].set_ylabel(ylabel='mmHg')
                axes[1].plot(
                    x_axis_values,
                    mini_batch_mask_as_weights[0],
                    label='ABP Mask'
                )
                axes[1].title.set_text(s="ABP Mask")
                axes[1].set_ylabel(ylabel='a. u.')
                show()

        if False:
            # computing the standatd deviation of the mean absolute error on the
            # validation set:
            _, validation_set, _ = create_datasets(mini_batch_size=1)
            predictions_mean_absolute_errors = []
            dbp_minimum_predictions_mean_absolute_errors = []
            sbp_maximum_predictions_mean_absolute_errors = []
            count = 0
            for mini_batch_data in validation_set:
                mini_batch_samples = mini_batch_data[0]
                mini_batch_labels = mini_batch_data[1]
                mini_batch_predictions = model.predict_on_batch(
                    x=mini_batch_samples
                ).squeeze()
                predictions_mean_absolute_errors.extend(
                    mean_absolute_error(
                        y_true=mini_batch_labels,
                        y_pred=mini_batch_predictions
                    ).numpy().tolist()
                )
                dbp_minimum_predictions_mean_absolute_errors.append(
                    mean_absolute_error(
                        y_true=min(mini_batch_labels),
                        y_pred=min(mini_batch_predictions)
                    ).numpy().tolist()
                )
                sbp_maximum_predictions_mean_absolute_errors.append(
                    mean_absolute_error(
                        y_true=max(mini_batch_labels),
                        y_pred=max(mini_batch_predictions)
                    ).numpy().tolist()
                )
                # save(
                #     file=f'mattia\\{count}.npy',
                #     arr=stack(
                #         arrays=[
                #             squeeze(a=mini_batch_labels),
                #             squeeze(a=mini_batch_predictions)
                #         ]
                #     )
                # )
                count += 1
            print(
                "MAE metric's mean over the validation set:",
                '{:.2f}'.format(
                    array(predictions_mean_absolute_errors).mean().item()
                )
            )
            print(
                "MAE metric's standard deviation over the validation set:",
                '{:.2f}'.format(
                    array(predictions_mean_absolute_errors).std().item()
                )
            )
            print(
                "DBP (as Minimum) MAE metric over the validation set:",
                '{:.2f}'.format(
                    array(
                        dbp_minimum_predictions_mean_absolute_errors
                    ).mean().item()
                )
            )
            print(
                "SBP (as Maximum) MAE metric over the validation set:",
                '{:.2f}'.format(
                    array(
                        sbp_maximum_predictions_mean_absolute_errors
                    ).mean().item()
                )
            )
            del mini_batch_samples
            del mini_batch_labels
            del predictions_mean_absolute_errors
            del validation_set
            raise Exception("This code branch should not have been run.")

    else:
        raise NotImplementedError("To be updated.")

        # letting the user choose which set to inspect beforehand, validating the
        # choice and setting variables for filenames and folders accordingly:
        chosen_set = input("What dataset do you want to inspect? [tr/val/test] ")
        if chosen_set == 'tr':
            set_dir = UCI_TRAINING_SET_DIR
            signal_block_segments_filenames = (
                UCI_TRAINING_SET_SIGNAL_BLOCK_SEGMENTS_FILENAMES
            )
        elif chosen_set == 'val':
            set_dir = UCI_VALIDATION_SET_DIR
            signal_block_segments_filenames = (
                UCI_VALIDATION_SET_SIGNAL_BLOCK_SEGMENTS_FILENAMES
            )
        elif chosen_set == 'test':
            set_dir = UCI_TEST_SET_DIR
            signal_block_segments_filenames = (
                UCI_TEST_SET_SIGNAL_BLOCK_SEGMENTS_FILENAMES
            )
        else:
            raise Exception("Unexpected choice.")

        while True:
            # extracting the input signals and the target signal of the signal
            # block segment chosen by the user (or a random one in case the user
            # is not willing to choose):
            signal_block_segment_name = input(
                "Filename of the signal segment of interest (leave it empty by " +
                "just pressing ENTER to let a random one be chosen for you): "
            )
            if signal_block_segment_name == '':
                # choosing a random signal block segment:
                signal_block_segment_name = choice(
                    seq=signal_block_segments_filenames
                )
            else:
                if (
                        signal_block_segment_name not in
                        signal_block_segments_filenames
                ):
                    raise Exception("Filename not found in the chosen set.")
            encoded_signal_block_segment = fromfile(
                file=path_join(
                    set_dir,
                    signal_block_segment_name
                ),
                dtype='float32'
            )
            assert encoded_signal_block_segment.shape[0] % 3 == 0
            input_signals = stack(
                arrays=(
                    encoded_signal_block_segment[0::3],
                    encoded_signal_block_segment[2::3]
                ),
                axis=-1
            )
            target_signal = encoded_signal_block_segment[1::3]

            # estimating the target signal from the input signals via model
            # prediction:
            predicted_signal = model.predict_on_batch(
                x=convert_to_tensor(
                    value=[input_signals],
                    dtype='float32'
                )
            ).squeeze()

            # comparing the predicted and the target signals by visually plotting
            # them, together with the input signals:
            figure, axes = subplots(3, sharex=True)
            figure.suptitle(
                t=f'Signal Block Segment from File "{signal_block_segment_name}"'
            )
            x_axis_values = (
                array(list(range(target_signal.shape[0]))) /
                SIGNALS_SAMPLING_FREQUENCY
            )
            axes[0].plot(x_axis_values, target_signal, label='target')
            axes[0].plot(x_axis_values, predicted_signal, label='predicted')
            axes[0].legend()
            axes[0].title.set_text(s="ABP")
            axes[0].set_ylabel(ylabel='mmHg')
            axes[1].plot(x_axis_values, input_signals[:, 0])
            axes[1].title.set_text(s="PPG")
            axes[1].set_ylabel(ylabel='adimensional')
            axes[2].plot(x_axis_values, input_signals[:, 1])
            axes[2].title.set_text(s="ECG (II Lead)")
            axes[2].set_ylabel(ylabel='mV')
            axes[2].set_xlabel(xlabel=str(float(mean_absolute_error(target_signal, predicted_signal))))#"seconds")
            show()

            # TODO: sort this out
            # from tensorflow.keras.utils import plot_model
            # plot_model(
            #     model,
            #     to_file="model.png",
            #     expand_nested=True,
            #     show_shapes=True,
            #     show_layer_activations=True
            # )

            # prompting to choose whether to try again with another signal:
            if input("\nDo you want to try again? [yes/no] ") == 'no':
                break
