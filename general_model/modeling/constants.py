"""
Common constant definitions.
"""


from argparse import ArgumentParser
from os import getcwd
from os.path import join as path_join
from random import seed as random_seed
from typing import get_args, Literal, Optional, Tuple

from numpy.random import rand, seed as numpy_seed
from scipy.signal import spectrogram
# pylint: disable=no-name-in-module, import-error
from tensorflow import expand_dims, float32 as tf_float32, squeeze
from tensorflow.random import set_seed
# pylint: enable=no-name-in-module, import-error
from yaml import safe_load as yaml_safe_load

from metrics import mean_signed_error


# ------------------------------- data aspects -------------------------------

CHANNEL_INDEX_FOR_ABP = 1
CHANNEL_INDEX_FOR_ECG = 2
CHANNEL_INDEX_FOR_ECG_MAGNITUDE = 2
CHANNEL_INDEX_FOR_ECG_PHASE = 4
CHANNEL_INDEX_FOR_PPG = 0
CHANNEL_INDEX_FOR_PPG_MAGNITUDE = 0
CHANNEL_INDEX_FOR_PPG_PHASE = 3

# data types:
DATA_TYPE_FOR_INPUTS = tf_float32
DATA_TYPE_FOR_OUTPUTS = tf_float32

# paths:

ROOT_DIR = getcwd()
DATA_DIR_NAME = 'data'
KERNELS_DIR_NAME = 'kernels'
IN_HOUSE_DATASET_ANNOTATIONS_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'preprocessed_in_house_dataset_annotations'
)
IN_HOUSE_DATASET_PARTITIONING_CONFIGURATIONS_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME
)
IN_HOUSE_DATASET_STATISTICS_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'preprocessed_in_house_dataset_statistics'
)
ORIGINAL_IN_HOUSE_DATASET_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'original_in_house_dataset'
)
ORIGINAL_UCI_DATASET_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'original_uci_dataset'
)
ORIGINAL_UCI_DATASET_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip'
)
PATH_TO_KERNEL_FOR_ABP = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    KERNELS_DIR_NAME,
    'abp_kernel.npy'
)
PATH_TO_KERNEL_FOR_ECG_LEAD = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    KERNELS_DIR_NAME,
    'ecg_lead_kernel.npy'
)
PATH_TO_KERNEL_FOR_PPG = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    KERNELS_DIR_NAME,
    'ppg_kernel.npy'
)
PREPROCESSED_IN_HOUSE_DATASET_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'preprocessed_in_house_dataset'
)
PREPROCESSED_UCI_DATASET_DIR = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'preprocessed_uci_dataset'
)
SIGNAL_EXAM_RESULTS_PATH = path_join(
    ROOT_DIR,
    DATA_DIR_NAME,
    'signals_acceptance_results.yaml'
)
IN_HOUSE_TEST_SET_DIR = path_join(
    PREPROCESSED_IN_HOUSE_DATASET_DIR,
    'test_set'
)
IN_HOUSE_TRAINING_SET_DIR = path_join(
    PREPROCESSED_IN_HOUSE_DATASET_DIR,
    'training_set'
)
IN_HOUSE_VALIDATION_SET_DIR = path_join(
    PREPROCESSED_IN_HOUSE_DATASET_DIR,
    'validation_set'
)
UCI_TEST_SET_DIR = path_join(
    PREPROCESSED_UCI_DATASET_DIR,
    'test_set'
)
UCI_TRAINING_SET_DIR = path_join(
    PREPROCESSED_UCI_DATASET_DIR,
    'training_set'
)
UCI_VALIDATION_SET_DIR = path_join(
    PREPROCESSED_UCI_DATASET_DIR,
    'validation_set'
)
# files containing data:
UCI_TEST_SET_SIGNAL_FILES_PATH_PATTERN = path_join(
    UCI_TEST_SET_DIR,
    '*.bin'
)
UCI_TRAINING_SET_SIGNAL_FILES_PATH_PATTERN = path_join(
    UCI_TRAINING_SET_DIR,
    '*.bin'
)
UCI_VALIDATION_SET_SIGNAL_FILES_PATH_PATTERN = path_join(
    UCI_VALIDATION_SET_DIR,
    '*.bin'
)

SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ = 1000
SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ = 400

N_IN_HOUSE_INPUT_CHANNELS = 16

N_SECONDS_PER_IN_HOUSE_SEGMENT = 10
N_SAMPLES_PER_IN_HOUSE_SEGMENT = round(
    N_SECONDS_PER_IN_HOUSE_SEGMENT *
    SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
)

N_UCI_SIGNAL_SEGMENT_SAMPLES = 960

NAME_FOR_IN_HOUSE_ABP_MASK = 'abp_mask'
NAME_FOR_IN_HOUSE_FINGER_ABP = 'finger_abp'
NAME_FOR_IN_HOUSE_BRACHIAL_ABP = 'brachial_abp'
NAME_FOR_IN_HOUSE_ACC_XYZ_MAGN = 'xyz_acc_magn'
NAME_FOR_IN_HOUSE_ACC_X = 'x_acc'
NAME_FOR_IN_HOUSE_ACC_Y = 'y_acc'
NAME_FOR_IN_HOUSE_ACC_Z = 'z_acc'
NAME_FOR_IN_HOUSE_ACOUSTIC = 'acoustic'
NAME_FOR_IN_HOUSE_BIOIMPEDANCE = 'bioimpedance'
NAME_FOR_IN_HOUSE_ECG = 'ecg'
NAME_FOR_IN_HOUSE_METADATA = 'metadata'
NAME_FOR_IN_HOUSE_PPG_GREEN = 'ppg_green'
NAME_FOR_IN_HOUSE_PPG_INFRARED = 'ppg_infrared'
NAME_FOR_IN_HOUSE_PPG_RED = 'ppg_red'
NAME_FOR_IN_HOUSE_RESP_PT_1 = 'resp_pt_1'
NAME_FOR_IN_HOUSE_RESP_PT_2 = 'resp_pt_2'
NAME_FOR_IN_HOUSE_RESP_PT_3 = 'resp_pt_3'
NAME_FOR_IN_HOUSE_SCG = 'scg'
NAME_FOR_IN_HOUSE_TEMP_BODY = 'body_temp'
NAME_FOR_IN_HOUSE_TEMP_EXT = 'ext_temp'

# dataset splitting amounts [relative amounts out of total]:
PORTION_OF_DATA_FOR_TEST = 0.1
PORTION_OF_DATA_FOR_VALIDATION = 0.2
PORTION_OF_DATA_FOR_TRAINING = 1 - (
    PORTION_OF_DATA_FOR_VALIDATION + PORTION_OF_DATA_FOR_TEST
)

SIGNALS_SAMPLING_FREQUENCY = 125  # [Hz]

SPECTROGRAM_SAMPLES_FOR_EACH_SFTF = 64  # 32
SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF = 32  # 15

(
    SPECTROGRAM_FREQUENCIES,
    SPECTROGRAM_SAMPLES,
    _
) = spectrogram(
    x=rand(N_UCI_SIGNAL_SEGMENT_SAMPLES),
    fs=SIGNALS_SAMPLING_FREQUENCY,
    nperseg=SPECTROGRAM_SAMPLES_FOR_EACH_SFTF,
    noverlap=SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF
    # NOTE: this spectrogram is based on real values of power
    # spectral density and with no redundant information repeated
    # on negative frequencies
)
N_SPECTROGRAM_FREQUENCIES = SPECTROGRAM_FREQUENCIES.shape[0]
N_SPECTROGRAM_SAMPLES = SPECTROGRAM_SAMPLES.shape[0]

N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES = N_UCI_SIGNAL_SEGMENT_SAMPLES - (
    N_SPECTROGRAM_FREQUENCIES * N_SPECTROGRAM_SAMPLES
)

# preprocessing parameters:
ECG_BANDPASS_FREQUENCIES = [0.5, 60]  # [Hz]
ECG_POWER_LINE_INTERFERENCE_FREQUENCY = 50  # [Hz]
THRESHOLD_FOR_PATHOLOGICALLY_HIGH_DBP = 130  # [mmHg]
THRESHOLD_FOR_PATHOLOGICALLY_HIGH_SBP = 180  # [mmHg]
THRESHOLD_FOR_PATHOLOGICALLY_LOW_DBP = 60  # [mmHg]
THRESHOLD_FOR_PATHOLOGICALLY_LOW_SBP = 80  # [mmHg]

# ----------------------------- modeling aspects -----------------------------

# paths:

MODELING_DIR_NAME = 'modeling'
RESULTS_DIR_NAME = 'results'
# data processing results cache:
CACHE_FILE_PATH_FOR_PROCESSED_DATASETS = path_join(
    ROOT_DIR,
    MODELING_DIR_NAME,
    'cache',
    'processed.tmp'
)
HYPERPARAMETERS_FILE_PATH = path_join(
    ROOT_DIR,
    MODELING_DIR_NAME,
    'hyperparameters.yaml'
)
LABELS_DISTRIBUTIONS_FITTING_PARAMETERS_FILE = path_join(
    ROOT_DIR,
    MODELING_DIR_NAME,
    'labels_distributions_fitting_parameters.yaml'
)

with open(HYPERPARAMETERS_FILE_PATH, 'r') as file:
    hyperparameters = yaml_safe_load(stream=file)

# architecture hyperparameters:
architecture_hyperparameters = hyperparameters['architecture']
ACTIVATION_FUNCTION = architecture_hyperparameters['activation_function']
BATCH_NORMALIZATION = architecture_hyperparameters['normalization']['layers']['batch_norm']
BATCH_NORMALIZATION_AXIS = -1
DOUBLE_UNET_VARIANT = architecture_hyperparameters['double_unet_variant']
DOUBLE_UNET_VARIANT_SUBMODEL_NAMES = [
    'OriginalUNet',
    'MultiResUNet'
]
DOWN_AND_UP_SAMPLING_SPAN = architecture_hyperparameters['down_and_up_sampling_span']
FREQUENCY_BIN_AVERAGING_DOWNSAMPLING_SPAN = 6
INITIALLY_SEPARATE_BRANCHES_IN_ENCODER_ONLY_MODEL = True
MULTI_RES_UNET_GRU_AFTER_BOTTLENECK = architecture_hyperparameters['multi_res_unet']['gru']['after_bottleneck']
MULTI_RES_UNET_GRU_AFTER_ENCODER = architecture_hyperparameters['multi_res_unet']['gru']['after_encoder']
MULTI_RES_UNET_GRU_AFTER_DECODER = architecture_hyperparameters['multi_res_unet']['gru']['after_decoder']
MULTI_RES_UNET_GRU_ALONG_SKIP_CONNECTIONS = architecture_hyperparameters['multi_res_unet']['gru']['along_skip_connections']
MULTI_RES_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK = architecture_hyperparameters['multi_res_unet']['n_convolutional_layers_per_block']
MULTI_RES_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS = architecture_hyperparameters['multi_res_unet']['n_convolutional_encoder_blocks']
MULTI_RES_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER = architecture_hyperparameters['multi_res_unet']['n_filters_in_first_encoder_layer']
MULTI_RES_UNET_SKIP_CONNECTIONS = architecture_hyperparameters['multi_res_unet']['skip_connections']
if MULTI_RES_UNET_GRU_ALONG_SKIP_CONNECTIONS:
    assert MULTI_RES_UNET_SKIP_CONNECTIONS, (
        "Enabled GRU along skip connections while skip connections disabled!"
    )
N_DENSE_WEIGHTS_DROP_FACTOR = 2
N_FILTERS_MULTIPLICATION_FACTOR = architecture_hyperparameters['n_filters_multiplication_factor']
N_FREQUENCY_BIN_AVERAGING_STEPS = 0
N_FULLY_CONNECTED_LAYERS = 6
N_WEIGHTS_IN_FIRST_ENCODER_LAYER = 2**12
VANILLA_UNET_GRU_AFTER_BOTTLENECK = architecture_hyperparameters['vanilla_unet']['gru']['after_bottleneck']
VANILLA_UNET_GRU_AFTER_ENCODER = architecture_hyperparameters['vanilla_unet']['gru']['after_encoder']
VANILLA_UNET_GRU_AFTER_DECODER = architecture_hyperparameters['vanilla_unet']['gru']['after_decoder']
VANILLA_UNET_GRU_ALONG_SKIP_CONNECTIONS = architecture_hyperparameters['vanilla_unet']['gru']['along_skip_connections']
VANILLA_UNET_MULTI_BRANCH_ENCODER = architecture_hyperparameters['vanilla_unet']['multi_branch_encoder']
VANILLA_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK = architecture_hyperparameters['vanilla_unet']['n_convolutional_layers_per_block']
VANILLA_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS = architecture_hyperparameters['vanilla_unet']['n_convolutional_encoder_blocks']
VANILLA_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER = architecture_hyperparameters['vanilla_unet']['n_filters_in_first_encoder_layer']
VANILLA_UNET_SKIP_CONNECTIONS = architecture_hyperparameters['vanilla_unet']['skip_connections']
if VANILLA_UNET_GRU_ALONG_SKIP_CONNECTIONS:
    assert VANILLA_UNET_SKIP_CONNECTIONS, (
        "Enabled GRU along skip connections while skip connections disabled!"
    )

# N_FILTERS_IN_FIRST_DECODER_LAYER = (
#     N_FILTERS_IN_FIRST_ENCODER_LAYER * (
#         N_FILTERS_MULTIPLICATION_FACTOR **
#         N_CONVOLUTIONAL_ENCODER_BLOCKS
#     )
# )
N_FILTERS_IN_FIRST_DECODER_LAYER = 2**9

# training hyperparameters:
training_hyperparameters = hyperparameters['training']
L2_REGULARIZATION_LAMBDA = training_hyperparameters['l2_regularization_lambda']
LEARNING_RATE_INITIAL_VALUE = training_hyperparameters['learning_rate']['initial_value']
LEARNING_RATE_FINAL_VALUE = training_hyperparameters['learning_rate']['final_or_constant_value']
LEARNING_RATE_TREND_ENABLED = training_hyperparameters['learning_rate']['trend_enabled']
MINI_BATCH_SIZE = training_hyperparameters['mini_batch_size']
N_EPOCHS_TO_REACH_IN_TOTAL = training_hyperparameters['n_epochs']
optimizer_hyperparameters = training_hyperparameters['optimizer']
OPTIMIZER_ID = None
n_optimizers_enabled = 0
for optimizer_id, optimizer_settings in optimizer_hyperparameters.items():
    if optimizer_settings['enabled']:
        n_optimizers_enabled += 1
        OPTIMIZER_ID = optimizer_id
assert n_optimizers_enabled == 1, "One and only one optimizer must be enabled."
OPTIMIZER_PARAMETERS = optimizer_hyperparameters[OPTIMIZER_ID]['parameters']

ValidArgumentsForOptimizerId = Literal['adam', 'sgd']
(
    OPTIMIZER_ID_FOR_ADAM,
    OPTIMIZER_ID_FOR_SGD
) = get_args(tp=ValidArgumentsForOptimizerId)

assert OPTIMIZER_ID in get_args(tp=ValidArgumentsForOptimizerId), (
    "Invalid optimizer identifier."
)

OUTPUT_NAME_FOR_DBP = 'dbp'
OUTPUT_NAME_FOR_SBP = 'sbp'

# hyperparameter tuning configurations:
METRIC_OF_INTEREST_NAME = 'metric_of_interest'
N_HYPERPARAMETER_COMBINATIONS_SAMPLED = 10

CUSTOM_OBJECTS_IN_MODEL_DEFINITION = {
    'expand_dims': expand_dims,
    'mean_signed_error': mean_signed_error,
    'squeeze': squeeze
}

INPUTS_STANDARDIZATION_LAYER_NAME = 'inputs_standardization'

# ----------------------------- reproducibility -----------------------------

STOCHASTIC_OPERATIONS_SEED = 0

# --------------------------------- settings ---------------------------------

ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP = 'minmax'
ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM = 'waveform'

ValidArgumentsForDataSource = Literal['in-house', 'uci']
ValidArgumentsForInputDomain = Literal['frequency', 'time', 'time-frequency']
ValidArgumentsForTask = Literal['minmax', 'waveform']

(
    ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE,
    ARGUMENT_NAME_FOR_DATA_SOURCE_AS_UCI
) = get_args(tp=ValidArgumentsForDataSource)
(
    ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY
) = get_args(tp=ValidArgumentsForInputDomain)
(
    ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP,
    ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM
) = get_args(tp=ValidArgumentsForTask)

# interactive CLI prompts:
SEQUENCE_BUILDING_MESSAGE = (
    "Building a new sequence of signal samples:"
)
SEQUENCE_EXPANSION_PROMPT_MESSAGE = (
    "Do you want to add another signal sample? [yes/no] "
)

# ----------------------------------------------------------------------------


def fix_seeds_for_reproducible_results() -> None:
    """
    Make the subsequent instructions produce purely deterministic outputs by
    fixing all the relevant seeds.
    """
    random_seed(a=0)
    _ = numpy_seed(seed=0)
    set_seed(seed=0)


def get_required_arguments(
        input_domain: bool,
        task: bool,
        data_source: bool,
        experiment_name: bool
) -> Tuple[
        Optional[ValidArgumentsForInputDomain],
        Optional[ValidArgumentsForTask],
        Optional[ValidArgumentsForDataSource],
        Optional[str]
]:
    """
    Get the arguments specified as required, among the ones representing:
    - the input domain for signals representation
    - the task to be carried out between sample-wise (full-waveform) ABP
      regression and signal-wise DBP and SBP regression
    - the data source between UCI-derived and in-house
    """
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        '-do',
        '--domain',
        choices=[
            ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY,
            ARGUMENT_NAME_FOR_DOMAIN_AS_TIME,
            ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY
        ],
        required=input_domain,
        type=str,
        dest='input_domain'
    )

    argument_parser.add_argument(
        '-t',
        '--task',
        choices=[
            ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM,
            ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP
        ],
        required=task,
        type=str,
        dest='task'
    )

    argument_parser.add_argument(
        '-da',
        '--data',
        choices=[
            ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE,
            ARGUMENT_NAME_FOR_DATA_SOURCE_AS_UCI
        ],
        required=data_source,
        type=str,
        dest='data_source'
    )

    argument_parser.add_argument(
        '-n',
        '--name',
        required=experiment_name,
        type=str,
        dest='experiment_name'
    )

    parsed_args = argument_parser.parse_args()

    return (
        parsed_args.input_domain,
        parsed_args.task,
        parsed_args.data_source,
        parsed_args.experiment_name
    )
