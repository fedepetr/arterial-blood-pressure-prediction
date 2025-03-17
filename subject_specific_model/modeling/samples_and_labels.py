"""
Loadage, extraction and preprocessing of samples and labels to feed the model.
"""


from math import pi, sqrt
from os import listdir
from os.path import join as path_join
from typing import Tuple

# pylint: disable=no-name-in-module,import-error
from tensorflow import (
    bool as tf_bool,
    cast,
    convert_to_tensor,
    expand_dims,
    float32 as tf_float32,
    function as tf_function,
    reshape,
    stack,
    Tensor
)
from tensorflow.data import AUTOTUNE, Dataset, TFRecordDataset
from tensorflow.io import (
    FixedLenFeature,
    decode_raw,
    parse_single_example,
    read_file
)
from tensorflow.math import (
    exp,
    logical_not,
    reduce_min,
    reduce_max,
    scalar_mul,
    square,
    subtract,
    truediv
)
# pylint: enable=no-name-in-module,import-error
from yaml import safe_load as yaml_safe_load

from constants import (  # noqa: F401 pylint: disable=unused-import
    ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE,
    ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY,
    ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP,
    ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM,
    CACHE_FILE_PATH_FOR_PROCESSED_DATASETS,
    CHANNEL_INDEX_FOR_ABP,
    CHANNEL_INDEX_FOR_ECG,
    CHANNEL_INDEX_FOR_ECG_MAGNITUDE,
    CHANNEL_INDEX_FOR_ECG_PHASE,
    CHANNEL_INDEX_FOR_PPG,
    CHANNEL_INDEX_FOR_PPG_MAGNITUDE,
    CHANNEL_INDEX_FOR_PPG_PHASE,
    DATA_TYPE_FOR_INPUTS,
    DATA_TYPE_FOR_OUTPUTS,
    IN_HOUSE_TEST_SET_DIR,
    IN_HOUSE_TRAINING_SET_DIR,
    IN_HOUSE_VALIDATION_SET_DIR,
    LABELS_DISTRIBUTIONS_FITTING_PARAMETERS_FILE,
    MINI_BATCH_SIZE,
    N_SAMPLES_PER_IN_HOUSE_SEGMENT,
    N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES,
    N_SPECTROGRAM_FREQUENCIES,
    N_SPECTROGRAM_SAMPLES,
    NAME_FOR_IN_HOUSE_ABP_MASK,
    NAME_FOR_IN_HOUSE_FINGER_ABP,
    NAME_FOR_IN_HOUSE_BRACHIAL_ABP,
    NAME_FOR_IN_HOUSE_ACC_XYZ_MAGN,
    NAME_FOR_IN_HOUSE_ACC_X,
    NAME_FOR_IN_HOUSE_ACC_Y,
    NAME_FOR_IN_HOUSE_ACC_Z,
    NAME_FOR_IN_HOUSE_ACOUSTIC,
    NAME_FOR_IN_HOUSE_BIOIMPEDANCE,
    NAME_FOR_IN_HOUSE_ECG,
    NAME_FOR_IN_HOUSE_METADATA,
    NAME_FOR_IN_HOUSE_PPG_GREEN,
    NAME_FOR_IN_HOUSE_PPG_INFRARED,
    NAME_FOR_IN_HOUSE_PPG_RED,
    NAME_FOR_IN_HOUSE_RESP_PT_1,
    NAME_FOR_IN_HOUSE_RESP_PT_2,
    NAME_FOR_IN_HOUSE_RESP_PT_3,
    NAME_FOR_IN_HOUSE_SCG,
    NAME_FOR_IN_HOUSE_TEMP_BODY,
    NAME_FOR_IN_HOUSE_TEMP_EXT,
    OUTPUT_NAME_FOR_DBP,
    OUTPUT_NAME_FOR_SBP,
    STOCHASTIC_OPERATIONS_SEED,
    UCI_TEST_SET_SIGNAL_FILES_PATH_PATTERN,
    UCI_TRAINING_SET_SIGNAL_FILES_PATH_PATTERN,
    UCI_VALIDATION_SET_SIGNAL_FILES_PATH_PATTERN,
    get_required_arguments
)


INPUT_DOMAIN, TASK, DATA_SOURCE, EXPERIMENT_NAME = get_required_arguments(
    input_domain=True,
    task=True,
    data_source=True,
    experiment_name=True
)

# labels' distributions fitting parameters:
with open(LABELS_DISTRIBUTIONS_FITTING_PARAMETERS_FILE, 'r') as file:
    (
        DBP_GAUSSIAN_MEAN,
        DBP_GAUSSIAN_STD,
        SBP_GAUSSIAN_MEAN,
        SBP_GAUSSIAN_STD
    ) = yaml_safe_load(stream=file)


def create_datasets(mini_batch_size: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load raw signals representing the dataset of interest from text lines
    inside and across different text files of pre-existing training,
    validation and test folders as already batched and paired as input
    samples' feature vectors and labels (after reconstructing samples and
    respective labels from the raw signals) - while also caching and
    prefetching the processed datasets' elements.
    """
    # loading the whole dataset of signals as already split among a training,
    # a validation and a test set following the pre-existing folder structure:
    (
        training_signals_set,
        validation_signals_set,
        test_signals_set
    ) = load_datasets_of_signals()

    # turning, for each dataset, the signals into mini-batches of paired input
    # samples' feature vectors and labels - while caching and prefetching
    # the processed datasets' elements:
    return (
        make_signals_feedable(
            dataset=training_signals_set,
            n_samples_per_mini_batch=mini_batch_size
        ),
        make_signals_feedable(
            dataset=validation_signals_set,
            n_samples_per_mini_batch=mini_batch_size
        ),
        make_signals_feedable(
            dataset=test_signals_set,
            n_samples_per_mini_batch=mini_batch_size
        )
    )


@tf_function
def load_numpy_array_from_file(file_path: Tensor) -> Tensor:
    """
    Load the numpy array contained in the file represented by the input path.
    """
    return decode_raw(
        input_bytes=read_file(
            filename=file_path
        ),
        out_type=DATA_TYPE_FOR_INPUTS
    )


def make_signals_feedable(
        dataset: Dataset,
        n_samples_per_mini_batch: int
) -> Dataset:
    """
    Turn the signals of the given dataset into mini-batches of paired input
    signal samples and signal labels - while caching and prefetching the
    processed dataset elements, returning a new dataset containing them.
    """
    if DATA_SOURCE == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE:
        if INPUT_DOMAIN != ARGUMENT_NAME_FOR_DOMAIN_AS_TIME:
            raise NotImplementedError(
                "Different input domain than time not yet supported on "
                "in-house data."
            )

        samples_dataset = dataset.map(
            map_func=lambda signal: stack(
                values=[
                    # high-frequency channels:
                    signal[NAME_FOR_IN_HOUSE_ACOUSTIC],
                    # medium-frequency channels:
                    signal[NAME_FOR_IN_HOUSE_ACC_XYZ_MAGN],
                    signal[NAME_FOR_IN_HOUSE_ACC_X],
                    signal[NAME_FOR_IN_HOUSE_ACC_Y],
                    signal[NAME_FOR_IN_HOUSE_ACC_Z],
                    signal[NAME_FOR_IN_HOUSE_BIOIMPEDANCE],
                    signal[NAME_FOR_IN_HOUSE_ECG],
                    signal[NAME_FOR_IN_HOUSE_PPG_GREEN],
                    signal[NAME_FOR_IN_HOUSE_PPG_INFRARED],
                    signal[NAME_FOR_IN_HOUSE_PPG_RED],
                    signal[NAME_FOR_IN_HOUSE_SCG],
                    # low-frequency channels:
                    signal[NAME_FOR_IN_HOUSE_RESP_PT_1],
                    signal[NAME_FOR_IN_HOUSE_RESP_PT_2],
                    signal[NAME_FOR_IN_HOUSE_RESP_PT_3],
                    signal[NAME_FOR_IN_HOUSE_TEMP_BODY],
                    signal[NAME_FOR_IN_HOUSE_TEMP_EXT],
                ],
                axis=1
            ),
            deterministic=True
        )

        labels_dataset = dataset.map(
            map_func=lambda signal: signal[NAME_FOR_IN_HOUSE_BRACHIAL_ABP],
            deterministic=True
        )

        labels_weights_dataset = dataset.map(
            # transforming intensities of and inverting the mask so as to have
            # ABP signals' parts covered by ones when to be considered and
            # zeros when not:
            map_func=lambda signal: expand_dims(
                input=cast(
                    x=logical_not(
                        x=cast(
                            x=signal[NAME_FOR_IN_HOUSE_ABP_MASK],
                            dtype=tf_bool
                        )
                    ),
                    dtype=DATA_TYPE_FOR_OUTPUTS
                ),
                axis=-1
            ),
            deterministic=True
        )

        if TASK == ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP:
            raise NotImplementedError(
                "Signal segments with invalid ABP tracts should be excluded."
            )

            dbp_scalars_dataset = labels_dataset.map(
                map_func=lambda signal: reduce_min(
                    input_tensor=signal,
                    keepdims=True
                ),
                deterministic=True
            )
            sbp_scalars_dataset = labels_dataset.map(
                map_func=lambda signal: reduce_max(
                    input_tensor=signal,
                    keepdims=True
                ),
                deterministic=True
            )

            labels_dataset = Dataset.zip(
                datasets={
                    OUTPUT_NAME_FOR_DBP: dbp_scalars_dataset,
                    OUTPUT_NAME_FOR_SBP: sbp_scalars_dataset
                }
            )

            labels_weights_dataset = Dataset.zip(
                datasets={
                    OUTPUT_NAME_FOR_DBP: labels_weights_dataset,
                    OUTPUT_NAME_FOR_SBP: labels_weights_dataset
                }
            )
        
        return (
            Dataset
            .zip(
                datasets=(
                    samples_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    ),
                    labels_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    ),
                    labels_weights_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    )
                )
            )
            # # optimizing performances by caching end-results:
            # .cache(filename=CACHE_FILE_PATH_FOR_PROCESSED_DATASETS)
            # # FIXME: TF bug above!!
            # randomly shuffling the elements of the considered set so as to
            # present them always in different orders during successive set
            # iterations, i.e. epochs:
            .shuffle(
                buffer_size=2**10,
                reshuffle_each_iteration=True,
                seed=STOCHASTIC_OPERATIONS_SEED
            )
            # optimizing performances by pre-fetching final elements:
            .prefetch(buffer_size=AUTOTUNE)
        )

    # defining the slicing step for reconstructing signals as the number of
    # signals encoded in the file array:
    if INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY:
        slicing_step = 5
    else:
        slicing_step = 3

    samples_signals_dataset = dataset.map(
        map_func=lambda signal: stack(
            values=(
                [
                    signal[CHANNEL_INDEX_FOR_PPG_MAGNITUDE::slicing_step],
                    signal[CHANNEL_INDEX_FOR_PPG_PHASE::slicing_step],
                    signal[CHANNEL_INDEX_FOR_ECG_MAGNITUDE::slicing_step],
                    signal[CHANNEL_INDEX_FOR_ECG_PHASE::slicing_step]
                ] if INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY
                else (
                    [
                        reshape(
                            tensor=signal[
                                CHANNEL_INDEX_FOR_PPG:
                                (
                                    - N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES
                                    * slicing_step
                                ):
                                slicing_step
                            ],
                            shape=(
                                N_SPECTROGRAM_FREQUENCIES,
                                N_SPECTROGRAM_SAMPLES
                            )
                        ),
                        reshape(
                            tensor=signal[
                                CHANNEL_INDEX_FOR_ECG:
                                (
                                    - N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES
                                    * slicing_step
                                ):
                                slicing_step
                            ],
                            shape=(
                                N_SPECTROGRAM_FREQUENCIES,
                                N_SPECTROGRAM_SAMPLES
                            )
                        )
                    ] if
                    INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY
                    else [
                        signal[CHANNEL_INDEX_FOR_PPG::slicing_step],
                        signal[CHANNEL_INDEX_FOR_ECG::slicing_step]
                    ]
                )
            ),
            axis=(
                1 if not
                INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY
                else 2
            )
        ),
        deterministic=True
    )

    labels_signals_dataset = dataset.map(
        map_func=lambda signal: (
            signal[CHANNEL_INDEX_FOR_ABP::slicing_step]
        ),
        deterministic=True
    )

    if TASK != ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM:
        dbp_scalars_dataset = labels_signals_dataset.map(
            map_func=lambda signal: reduce_min(
                input_tensor=signal,
                keepdims=True
            ),
            deterministic=True
        )
        sbp_scalars_dataset = labels_signals_dataset.map(
            map_func=lambda signal: reduce_max(
                input_tensor=signal,
                keepdims=True
            ),
            deterministic=True
        )
        labels_scalars_dataset = Dataset.zip(
            datasets={
                OUTPUT_NAME_FOR_DBP: dbp_scalars_dataset,
                OUTPUT_NAME_FOR_SBP: sbp_scalars_dataset
            }
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # TODO: decide
        # labels_weights_dataset = Dataset.zip(
        #     datasets={
        #         OUTPUT_NAME_FOR_DBP: dbp_scalars_dataset.map(
        #             map_func=weight_dbp_labels,
        #             deterministic=True
        #         ),
        #         OUTPUT_NAME_FOR_SBP: sbp_scalars_dataset.map(
        #             map_func=weight_sbp_labels,
        #             deterministic=True
        #         )
        #     }
        # )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return (
        Dataset
        .zip(
            datasets=(
                (
                    samples_signals_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    ),
                    labels_signals_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    )
                ) if TASK == ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM else (
                    samples_signals_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    ),
                    labels_scalars_dataset.batch(
                        batch_size=n_samples_per_mini_batch,
                        drop_remainder=False,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=True
                    ),
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    # TODO: decide
                    # labels_weights_dataset.batch(
                    #     batch_size=n_samples_per_mini_batch,
                    #     drop_remainder=False,
                    #     num_parallel_calls=AUTOTUNE,
                    #     deterministic=True
                    # )
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                )
            )
        )
        # # optimizing performances by caching end-results:
        # .cache(filename=CACHE_FILE_PATH_FOR_PROCESSED_DATASETS)
        # # FIXME: TF bug above!!
        # randomly shuffling the elements of the considered set so as to
        # present them always in different orders during successive set
        # iterations, i.e. epochs:
        .shuffle(
            buffer_size=2**10,
            reshuffle_each_iteration=True,
            seed=STOCHASTIC_OPERATIONS_SEED
        )
        # optimizing performances by pre-fetching final elements:
        .prefetch(buffer_size=AUTOTUNE)
    )


def load_datasets_of_signals() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load elements (representing signals) of the whole dataset into a training
    set, a validation set and a test set dividing them following the
    pre-existing folder structure.
    """
    if DATA_SOURCE == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE:
        training_dataset = load_single_set_of_in_house_signals(
            tfrecords_dir=IN_HOUSE_TRAINING_SET_DIR
        )
        validation_dataset = load_single_set_of_in_house_signals(
            tfrecords_dir=IN_HOUSE_VALIDATION_SET_DIR
        )
        test_dataset = load_single_set_of_in_house_signals(
            tfrecords_dir=IN_HOUSE_TEST_SET_DIR
        )

    else:
        training_dataset = load_single_set_of_uci_signals(
            signal_files_path_pattern=(
                UCI_TRAINING_SET_SIGNAL_FILES_PATH_PATTERN
            )
        )
        validation_dataset = load_single_set_of_uci_signals(
            signal_files_path_pattern=(
                UCI_VALIDATION_SET_SIGNAL_FILES_PATH_PATTERN
            )
        )
        test_dataset = load_single_set_of_uci_signals(
            signal_files_path_pattern=(
                UCI_TEST_SET_SIGNAL_FILES_PATH_PATTERN
            )
        )

    return (training_dataset, validation_dataset, test_dataset)


def load_single_set_of_in_house_signals(tfrecords_dir: str) -> Dataset:
    """
    Loading a single set of signals saved to the file system as TF dataset.
    """
    features_description = {
        NAME_FOR_IN_HOUSE_METADATA: FixedLenFeature(
            shape=(1,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ABP_MASK: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_FINGER_ABP: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_BRACHIAL_ABP: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ACC_XYZ_MAGN: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ACC_X: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ACC_Y: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ACC_Z: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ACOUSTIC: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_BIOIMPEDANCE: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_ECG: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_PPG_RED: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_PPG_INFRARED: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_PPG_GREEN: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_RESP_PT_1: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_RESP_PT_2: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_RESP_PT_3: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_SCG: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_TEMP_BODY: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
        NAME_FOR_IN_HOUSE_TEMP_EXT: FixedLenFeature(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT,),
            dtype=tf_float32
        ),
    }

    def parse_tfrecord_features(inputs):
        return parse_single_example(
            serialized=inputs,
            features=features_description
        )

    tfrecord_filenames = []
    for filename in listdir(tfrecords_dir):
        if not filename.endswith('.tfrecord'):
            continue
        tfrecord_filenames.append(path_join(tfrecords_dir, filename))

    return TFRecordDataset(filenames=tfrecord_filenames).map(
        map_func=parse_tfrecord_features,
        deterministic=True
    )


def load_single_set_of_uci_signals(signal_files_path_pattern: str) -> Dataset:
    """
    Load elements (representing signals) of the set whose path pattern is
    given as input.
    """
    return (
        # creating a dataset of paths to the files containing the set signals:
        Dataset.list_files(
            file_pattern=signal_files_path_pattern,
            shuffle=False,
            seed=STOCHASTIC_OPERATIONS_SEED
        )
        # extracting the content of such files, each one as a signal:
        .map(
            map_func=load_numpy_array_from_file,
            deterministic=True
        )
    )


@tf_function
def weight_dbp_labels(labels: Tensor) -> Tensor:
    """
    Compute weights, for the input DBP labels, that are inversely proportional
    to their distribution on the training set.
    """
    return subtract(
        x=convert_to_tensor(value=1, dtype=DATA_TYPE_FOR_INPUTS),
        y=truediv(
            x=truediv(
                x=exp(
                    x=scalar_mul(
                        scalar=(-1/2),
                        x=square(
                            x=truediv(
                                x=subtract(
                                    x=cast(x=labels, dtype=DATA_TYPE_FOR_INPUTS),
                                    y=convert_to_tensor(
                                        value=DBP_GAUSSIAN_MEAN,
                                        dtype=DATA_TYPE_FOR_INPUTS
                                    )
                                ),
                                y=convert_to_tensor(
                                    value=DBP_GAUSSIAN_STD,
                                    dtype=DATA_TYPE_FOR_INPUTS
                                )
                            )
                        )
                    )
                ),
                y=convert_to_tensor(
                    value=(DBP_GAUSSIAN_STD * sqrt(2*pi)),
                    dtype=DATA_TYPE_FOR_INPUTS
                )
            ),
            y=convert_to_tensor(
                value=(DBP_GAUSSIAN_STD * sqrt(2*pi)),
                dtype=DATA_TYPE_FOR_INPUTS
            )
        )
    )


@tf_function
def weight_sbp_labels(labels: Tensor) -> Tensor:
    """
    Compute weights, for the input SBP labels, that are inversely proportional
    to their distribution on the training set.
    """
    return subtract(
        x=convert_to_tensor(value=1, dtype=DATA_TYPE_FOR_INPUTS),
        y=truediv(
            x=truediv(
                x=exp(
                    x=scalar_mul(
                        scalar=(-1/2),
                        x=square(
                            x=truediv(
                                x=subtract(
                                    x=cast(x=labels, dtype=DATA_TYPE_FOR_INPUTS),
                                    y=convert_to_tensor(
                                        value=SBP_GAUSSIAN_MEAN,
                                        dtype=DATA_TYPE_FOR_INPUTS
                                    )
                                ),
                                y=convert_to_tensor(
                                    value=SBP_GAUSSIAN_STD,
                                    dtype=DATA_TYPE_FOR_INPUTS
                                )
                            )
                        )
                    )
                ),
                y=convert_to_tensor(
                    value=(SBP_GAUSSIAN_STD * sqrt(2*pi)),
                    dtype=DATA_TYPE_FOR_INPUTS
                )
            ),
            y=convert_to_tensor(
                value=(SBP_GAUSSIAN_STD * sqrt(2*pi)),
                dtype=DATA_TYPE_FOR_INPUTS
            )
        )
    )


if __name__ == '__main__':
    training_set, validation_set, test_set = create_datasets(
        mini_batch_size=MINI_BATCH_SIZE
    )

    if INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME:
        if TASK == ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM:
            for mini_batch_data in training_set:
                mini_batch_samples = mini_batch_data[0]
                mini_batch_labels = mini_batch_data[1]
                print('—'*90)
                print(
                    f"mini-batch with {mini_batch_samples.shape[0]}"
                    f" samples of shape {mini_batch_samples.shape[1:]}"
                    f" and {mini_batch_labels.shape[0]} labels "
                    f"of shape {mini_batch_labels.shape[1:]}"
                )
                if len(mini_batch_data) == 3:
                    mini_batch_weights = mini_batch_data[2]
                    print('\t', mini_batch_weights.shape[0], mini_batch_weights.shape[1:])
        else:
            for mini_batch_data in training_set:
                mini_batch_samples = mini_batch_data[0]
                mini_batch_labels = mini_batch_data[1]
                print('—'*90)
                print(
                    f"mini-batch with {mini_batch_samples.shape[0]} "
                    f"samples of shape {mini_batch_samples.shape[1:]} "
                    f"and {len(mini_batch_labels)} label dictionaries "
                    "respectively with "
                    f"{mini_batch_labels[OUTPUT_NAME_FOR_DBP].shape[0]} "
                    "labels of shape "
                    f"{mini_batch_labels[OUTPUT_NAME_FOR_DBP].shape[1:]} "
                    "and with "
                    f"{mini_batch_labels[OUTPUT_NAME_FOR_SBP].shape[0]} "
                    "labels of shape "
                    f"{mini_batch_labels[OUTPUT_NAME_FOR_SBP].shape[1:]}",
                    end='\n\t'
                )
                if len(mini_batch_data) == 3:
                    mini_batch_weights = mini_batch_data[2]
                    print(
                        f"and {len(mini_batch_weights)} weight dictionaries "
                        "respectively with "
                        f"{mini_batch_weights[OUTPUT_NAME_FOR_DBP].shape[0]} "
                        "weights of shape "
                        f"{mini_batch_weights[OUTPUT_NAME_FOR_DBP].shape[1:]} "
                        "and with "
                        f"{mini_batch_weights[OUTPUT_NAME_FOR_SBP].shape[0]} "
                        "weights of shape "
                        f"{mini_batch_weights[OUTPUT_NAME_FOR_SBP].shape[1:]}"
                    )
