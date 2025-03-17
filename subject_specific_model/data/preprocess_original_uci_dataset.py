"""
Script to preprocess the downloaded original UCI dataset.
"""


from copy import deepcopy
from os import getcwd, listdir
from os.path import join as path_join
from random import choices, randint, seed as random_seed
from sys import path
from tkinter import font
from typing import List, Optional, Tuple
from yaml import dump as yaml_dump, load as yaml_load

from h5py import File
from matplotlib.pyplot import (
    close,
    hist,
    figure,
    pause,
    show,
    stairs,
    subplots,
    title,
    vlines,
    xticks
)
from numpy import (
    absolute,
    angle,
    arange,
    array,
    empty,
    histogram,
    isnan,
    load,
    ndarray,
    pad
)
from scipy.fft import fft
from scipy.signal import correlate, filtfilt, iirfilter, iirnotch, spectrogram
from scipy.stats import zscore

path.append(path_join(getcwd(), 'modeling'))
from constants import (
    ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY,
    CHANNEL_INDEX_FOR_ABP,
    CHANNEL_INDEX_FOR_ECG,
    CHANNEL_INDEX_FOR_ECG_MAGNITUDE,
    CHANNEL_INDEX_FOR_ECG_PHASE,
    CHANNEL_INDEX_FOR_PPG,
    CHANNEL_INDEX_FOR_PPG_MAGNITUDE,
    CHANNEL_INDEX_FOR_PPG_PHASE,
    ECG_BANDPASS_FREQUENCIES,
    ECG_POWER_LINE_INTERFERENCE_FREQUENCY,
    get_required_arguments,
    N_UCI_SIGNAL_SEGMENT_SAMPLES,
    ORIGINAL_UCI_DATASET_DIR,
    PATH_TO_KERNEL_FOR_ABP,
    PATH_TO_KERNEL_FOR_ECG_LEAD,
    PATH_TO_KERNEL_FOR_PPG,
    PORTION_OF_DATA_FOR_TEST,
    PORTION_OF_DATA_FOR_TRAINING,
    PORTION_OF_DATA_FOR_VALIDATION,
    SIGNAL_EXAM_RESULTS_PATH,
    SIGNALS_SAMPLING_FREQUENCY,
    N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES,
    SPECTROGRAM_SAMPLES_FOR_EACH_SFTF,
    SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF,
    UCI_TEST_SET_DIR,
    THRESHOLD_FOR_PATHOLOGICALLY_HIGH_DBP,
    THRESHOLD_FOR_PATHOLOGICALLY_HIGH_SBP,
    THRESHOLD_FOR_PATHOLOGICALLY_LOW_DBP,
    THRESHOLD_FOR_PATHOLOGICALLY_LOW_SBP,
    UCI_TRAINING_SET_DIR,
    UCI_VALIDATION_SET_DIR
)


FILTER_ARGS_FOR_ECG_BASELINE_WANDERING = {}
(
    FILTER_ARGS_FOR_ECG_BASELINE_WANDERING['b'],
    FILTER_ARGS_FOR_ECG_BASELINE_WANDERING['a']
) = iirfilter(
    N=4,
    Wn=ECG_BANDPASS_FREQUENCIES,
    btype='bandpass',
    ftype='bessel',
    fs=SIGNALS_SAMPLING_FREQUENCY
)
FILTER_ARGS_FOR_ECG_POWER_LINE_INTERFACE_NOTCH = {}
(
    FILTER_ARGS_FOR_ECG_POWER_LINE_INTERFACE_NOTCH['b'],
    FILTER_ARGS_FOR_ECG_POWER_LINE_INTERFACE_NOTCH['a']
) = iirnotch(
    w0=ECG_POWER_LINE_INTERFERENCE_FREQUENCY,
    Q=ECG_POWER_LINE_INTERFERENCE_FREQUENCY / 5,
    fs=SIGNALS_SAMPLING_FREQUENCY
)
INPUT_DOMAIN, _, _, _ = get_required_arguments(
    input_domain=True,
    task=False,
    data_source=False,
    experiment_name=False
)
# PRELIMINARY_INSPECTION = (
#     environ['PRELIMINARY_INSPECTION'] in ['True', 'true']
# )
PRELIMINARY_INSPECTION = False
with open(SIGNAL_EXAM_RESULTS_PATH, 'r') as file:
    SIGNALS_INDEX_TO_ACCEPTANCE = (
        yaml_load(stream=file)
    )
STANDARDIZED_KERNEL_FOR_ABP = zscore(a=load(PATH_TO_KERNEL_FOR_ABP))
STANDARDIZED_KERNEL_FOR_ECG_LEAD = zscore(a=load(PATH_TO_KERNEL_FOR_ECG_LEAD))
STANDARDIZED_KERNEL_FOR_PPG = zscore(a=load(PATH_TO_KERNEL_FOR_PPG))
STANDARDIZED_KERNEL_FOR_INVERTED_PPG = STANDARDIZED_KERNEL_FOR_PPG[::-1]


def plot_blocking_histogram(
        bin_counts: ndarray,
        bin_edges: ndarray,
        color: str,
        title_text: str
) -> None:
    """
    Plot a histogram, based on the input bin edges and counts, that blocks the
    current thread until closed.
    """
    figure()
    stairs(
        values=bin_counts,
        edges=bin_edges,
        fill=True,
        color=color
    )
    vlines(
        x=bin_edges,
        ymax=bin_counts.max(),
        ymin=0,
        colors='white'
    )
    title(label=title_text)
    show()


mattia = [
    # [3130, 30058],
    # [3188, 13121],
    [4784, 4673],
    [2397, 4427],
    [4703, 12950],
    [10354, 999],
    [9368, 8076],
    [11493, 1160],
    [540, 999],
    [3252, 20818],
    [5686, 999],
    [11765, 2013],
    [2864, 1053],
    [11765, 4852],
    [10817, 999],
    [8569, 3321],
    [8857, 999],
    [8852, 4328],
    [11501, 1027],
    [6422, 30134],
    [2684, 1305],
    [3332, 4803],
    [4104, 9724],
    [9335, 12102],
    [3498, 999],
    [11636, 999],
    [3909, 53647],
    [10324, 2372],
    [11964, 22311],
    [3130, 30058],
    [1314, 12456],
    [10708, 5265],
    [331, 3126],
    [6883, 17615],
    [11553, 40445],
    [1318, 999],
    [3188, 13121],
    [7382, 17142],
    [6095, 3792],
    [1018, 1205],
    [3621, 999],
    [7462, 2459],
    [2550, 8533],
    [7421, 999],
    [2407, 21324],
    [7339, 6976],
    [9497, 19035],
    [7649, 1101],
    [985, 10995],
    [5866, 29630],
    [4617, 5578],
    [10008, 27800],
    [10025, 33346],
    [4756, 4011],
    [6311, 2551],
    [1649, 999],
    [11318, 1814],
    [9431, 8932],
    [7983, 60508],
    [1641, 2947],
    [8166, 9168],
    [5458, 1248],
    [4961, 27742],
    [666, 999],
    [9324, 34330],
    [6590, 6748],
    [5382, 5901],
    [9832, 6003],
    [7246, 3725],
    [6789, 2032],
    [6306, 1648],
    [2038, 999],
    [6801, 13319],
    [1087, 30502],
    [8268, 19253],
    [5272, 3323],
    [11431, 23040],
    [1920, 40110],
    [8674, 5106],
]
c = 0
from matplotlib import rcParams
from matplotlib.pyplot import close, savefig
rcParams.update({'font.size': 18})
def plot_random_signal_block_parts_forever(
        whole_signal_blocks: List[ndarray],
        n_blocks: int,
        minimum_signal_length_ensured: int
) -> None:
    """
    Display random parts of random signal blocks in sequential plots each
    blocking the current thread until closed.
    """
    plotted_length = 1000 - 1
    assert plotted_length < minimum_signal_length_ensured

    while True:
        # getting a random segment of a random signal block:
        block_index = randint(a=0, b=(n_blocks - 1))
        block = whole_signal_blocks[block_index]
        block_length = block.shape[0]
        segment_initial_sample_index = randint(
            a=0,
            b=((block_length - plotted_length) - 1)
        )
        segment_final_sample_index = (
            segment_initial_sample_index + plotted_length
        )
        signals_segment = block[
            segment_initial_sample_index:segment_final_sample_index,
            :
        ]

        # defining x-axis values to plot as sample numbers:
        x_axis_values = list(range(signals_segment.shape[0]))

        if [block_index, segment_final_sample_index] in mattia:
            mattia.remove([block_index, segment_final_sample_index])
        else:
            continue
        # plotting the signal block segment:
        title = (
            f"Signal Block {block_index}, " +
            f"Samples {segment_initial_sample_index} " +
            f"- {segment_final_sample_index} "
        )
        figure, axes = subplots(3, sharex=True, figsize=(6.4 * 2, 4.8 * 2))
        figure.suptitle(t=title)
        x_axis_values = array(x_axis_values) / SIGNALS_SAMPLING_FREQUENCY
        axes[0].plot(x_axis_values, signals_segment[:, CHANNEL_INDEX_FOR_PPG])
        axes[0].title.set_text(s="PPG")
        axes[0].set_ylabel(ylabel="adimensional")
        axes[1].plot(x_axis_values, signals_segment[:, CHANNEL_INDEX_FOR_ECG])
        axes[1].title.set_text(s="ECG (II Lead)")
        axes[1].set_ylabel(ylabel="mV")
        axes[2].plot(x_axis_values, signals_segment[:, CHANNEL_INDEX_FOR_ABP])
        axes[2].title.set_text(s="ABP")
        axes[2].set_ylabel(ylabel="mmHg")
        axes[2].set_xlabel(xlabel="seconds")
        # show()
        global c; c += 1
        savefig("C:\\Users\\Mattia\\Desktop\\ABP prediction\\source\\plots\\Dataset's Low Quality\\new - correct size, time axis and font size\\" + str(c) + '.png',)
        if mattia == []:
            raise Exception('done')
        close()


def preprocess_signal_block_segment(
        signal_block_segment: ndarray,
        segmentwise_inputs_standardization: bool,
        plot_changes: bool
) -> Tuple[Optional[ndarray], bool]:
    """
    Preprocess the input signal block segment by normalizing its PPG component
    amplitude and its ECG component amplitude so that each one of the two
    signal segments has mean zero and unitary standard deviation, while also
    returning a flag to discard the signal block when some anomalies are
    detacted, such as when ABP is pathologically high/low, when ECG is
    collected from an unexpected lead, when PPG is inverted since acquired in
    an unexpected mode (reflection vs trasmittance).
    """
    returned_outputs_when_to_be_discarded = (None, True)

    if plot_changes:
        discarded = False
        original_signal_block_segment = deepcopy(x=signal_block_segment)

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ABP ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    maximum_abp_reached = signal_block_segment[:, CHANNEL_INDEX_FOR_ABP].max()
    mimimum_abp_reached = signal_block_segment[:, CHANNEL_INDEX_FOR_ABP].min()

    # discarding the segment if ...
    if (
        # ... either its DBP touches pathologically high values:
        mimimum_abp_reached >= THRESHOLD_FOR_PATHOLOGICALLY_HIGH_DBP
        # ... or its SBP touches pathologically high values:
        or maximum_abp_reached >= THRESHOLD_FOR_PATHOLOGICALLY_HIGH_SBP
        # ... or its DBP touches pathologically low values:
        or mimimum_abp_reached <= THRESHOLD_FOR_PATHOLOGICALLY_LOW_DBP
        # ... or its SBP touches pathologically low values:
        or maximum_abp_reached <= THRESHOLD_FOR_PATHOLOGICALLY_LOW_SBP
        # ... or its ABP wavewform is distorted to a degree over the defined
        # acceptance threshold (by comparison of standardized
        # cross-correlation maximum with an ideal ABP cicle kernel):
        or (
            correlate(
                zscore(a=signal_block_segment[:, CHANNEL_INDEX_FOR_ABP]),
                STANDARDIZED_KERNEL_FOR_ABP,
                mode='same',
                method='direct'
            ).max() < 50
        )
    ):
        if not plot_changes:
            return returned_outputs_when_to_be_discarded
        else:
            discarded = True

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ECG ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    # filtering the ECG signal to remove baseline wandering:
    signal_block_segment[:, CHANNEL_INDEX_FOR_ECG] = filtfilt(
        x=signal_block_segment[:, CHANNEL_INDEX_FOR_ECG],
        **FILTER_ARGS_FOR_ECG_BASELINE_WANDERING
    )

    # filtering the ECG signal to remove power line interference:
    signal_block_segment[:, CHANNEL_INDEX_FOR_ECG] = filtfilt(
        x=signal_block_segment[:, CHANNEL_INDEX_FOR_ECG],
        **FILTER_ARGS_FOR_ECG_POWER_LINE_INTERFACE_NOTCH
    )

    # discarding the segment if its ECG does not have any strong enough peak
    # in the normalized cross-correlation with a representative kernel of the
    # desired ECG lead pattern:
    if correlate(
            zscore(a=signal_block_segment[:, CHANNEL_INDEX_FOR_ECG]),
            STANDARDIZED_KERNEL_FOR_ECG_LEAD,
            mode='same',
            method='direct'
    ).max() < 50:
        if not plot_changes:
            return returned_outputs_when_to_be_discarded
        else:
            discarded = True

    if segmentwise_inputs_standardization:
        # ECG standardization:
        signal_block_segment[:, CHANNEL_INDEX_FOR_ECG] = zscore(
            a=signal_block_segment[:, CHANNEL_INDEX_FOR_ECG],
            nan_policy='raise'
        )

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PPG ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    # filtering PPG baseline wandering out:
    # NOTE: not done as it may be informative

    # discarding segments with inverted PPG mode (transmission vs reflectance):
    if (
            correlate(
                zscore(a=signal_block_segment[:, CHANNEL_INDEX_FOR_PPG]),
                STANDARDIZED_KERNEL_FOR_PPG,
                mode='same',
                method='direct'
            ).max() < correlate(
                zscore(a=signal_block_segment[:, CHANNEL_INDEX_FOR_PPG]),
                STANDARDIZED_KERNEL_FOR_INVERTED_PPG,
                mode='same',
                method='direct'
            ).max()
    ):
        if not plot_changes:
            return returned_outputs_when_to_be_discarded
        else:
            discarded = True

    if segmentwise_inputs_standardization:
        # PPG standardization:
        signal_block_segment[:, CHANNEL_INDEX_FOR_PPG] = zscore(
            a=signal_block_segment[:, CHANNEL_INDEX_FOR_PPG],
            nan_policy='raise'
        )

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    if not plot_changes:
        return signal_block_segment, False

    else:
        figure, axes = subplots(3, sharex=True)

        figure.suptitle(
            t=f'{"Discarded" if discarded else "Kept"}'
        )

        x_axis_values = list(range(N_UCI_SIGNAL_SEGMENT_SAMPLES))

        axes[0].plot(
            x_axis_values,
            original_signal_block_segment[:, CHANNEL_INDEX_FOR_PPG],
            color='deepskyblue',
            linestyle='dashdot',
            label='original'
        )
        axes[0].plot(
            x_axis_values,
            signal_block_segment[:, CHANNEL_INDEX_FOR_PPG],
            color='lawngreen',
            linestyle='dotted',
            label='preprocessed'
        )
        axes[0].title.set_text(s="PPG")
        axes[0].legend()

        axes[1].plot(
            x_axis_values,
            original_signal_block_segment[:, CHANNEL_INDEX_FOR_ECG],
            color='deepskyblue',
            linestyle='dashdot',
            label='original'
        )
        axes[1].plot(
            x_axis_values,
            signal_block_segment[:, CHANNEL_INDEX_FOR_ECG],
            color='lawngreen',
            linestyle='dotted',
            label='preprocessed'
        )
        axes[1].title.set_text(s="ECG")
        axes[1].legend()

        axes[2].plot(
            x_axis_values,
            original_signal_block_segment[:, CHANNEL_INDEX_FOR_ABP],
            color='deepskyblue',
            linestyle='dashdot',
            label='original'
        )
        axes[2].plot(
            x_axis_values,
            signal_block_segment[:, CHANNEL_INDEX_FOR_ABP],
            color='lawngreen',
            linestyle='dotted',
            label='preprocessed'
        )
        axes[2].title.set_text(s="ABP")
        axes[2].legend()

        show(block=False)
        pause(interval=5)  # in seconds
        close()

        return (
            returned_outputs_when_to_be_discarded if discarded else
            (signal_block_segment, False)
        )


def segment_signal_block_and_filter_segments(
        whole_length_signal_block: ndarray,
        segment_length_in_samples: int,
        standardize_inputs_segmentwise: bool,
        save: bool,
        base_file_name: Optional[str] = None
) -> int:
    """
    Segment the given whole-length signal block according to the specified
    length (intended as number of samples), filter it so as to normalize its
    PPG component amplitude and optionally save the resulting segments to the
    file system taking the input base name as file name prefix. The number of
    signal block segments successfully considered is returned.
    """
    total_length_in_samples = whole_length_signal_block.shape[0]

    n_good_segments = 0

    # for each length segment:
    for segment_starting_sample_index in range(
            0, total_length_in_samples, segment_length_in_samples
    ):
        # extracting the respective signal block segment:
        next_segment_starting_sample_index = (
            segment_starting_sample_index + segment_length_in_samples
        )
        # TODO: use masking and padding instead of discarding signal portions:
        # --------------------------------------------------------------------
        if next_segment_starting_sample_index > total_length_in_samples:
            continue
        segment = whole_length_signal_block[
            segment_starting_sample_index:next_segment_starting_sample_index,
            :
        ]
        # segment = whole_length_signal_block[
        #     segment_starting_sample_index:(
        #         next_segment_starting_sample_index if (
        #             next_segment_starting_sample_index <
        #             total_length_in_samples
        #         ) else -1
        #     )
        # ]
        # --------------------------------------------------------------------

        # preprocessing the current signal block segment components used as
        # inputs by separately standardizing their respective amplitudes:
        segment, to_be_discarded = preprocess_signal_block_segment(
            signal_block_segment=segment,
            segmentwise_inputs_standardization=standardize_inputs_segmentwise,
            plot_changes=(not save)
        )

        # discarding the signal segment if some anomalies are detected:
        if to_be_discarded:
            continue

        # ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
        # plotting the same input across different domains:

        # figure, axes = subplots(2, sharex=True, figsize=(6.4*2, 4.8*2))
        # figure.suptitle(t="Time-Domain Input Representation")
        # x_axis_values = array(list(range(N_UCI_SIGNAL_SEGMENT_SAMPLES))) / SIGNALS_SAMPLING_FREQUENCY
        # axes[0].plot(x_axis_values, segment[:, CHANNEL_INDEX_FOR_PPG])
        # axes[0].title.set_text(s="PPG")
        # axes[0].set_ylabel(ylabel="a.u.")
        # axes[1].plot(x_axis_values, segment[:, CHANNEL_INDEX_FOR_ECG])
        # axes[1].title.set_text(s="ECG")
        # axes[1].set_ylabel(ylabel="mV")
        # axes[1].set_xlabel(xlabel="seconds")
        # show()

        # from matplotlib.pyplot import subplots_adjust
        # from scipy.fft import fftfreq, fftshift
        # ppg_fft = fftshift(x=fft(x=segment[:, CHANNEL_INDEX_FOR_PPG]))
        # ecg_fft = fftshift(x=fft(x=segment[:, CHANNEL_INDEX_FOR_ECG]))
        # figure, axes = subplots(4, sharex=True, figsize=(6.4*2, 4.8*2))
        # subplots_adjust(hspace=0.5)
        # figure.suptitle(t="Frequency-Domain Input Representation")
        # x_axis_values = fftshift(x=fftfreq(n=N_UCI_SIGNAL_SEGMENT_SAMPLES, d=1/SIGNALS_SAMPLING_FREQUENCY))
        # # import pdb; pdb.set_trace()
        # axes[0].plot(x_axis_values, absolute(ppg_fft))
        # axes[0].title.set_text(s="PPG Magnitude")
        # axes[0].set_ylabel(ylabel="a.u.")
        # axes[1].plot(x_axis_values, angle(ppg_fft))
        # axes[1].title.set_text(s="PPG Phase")
        # axes[1].set_ylabel(ylabel="radians")
        # axes[2].plot(x_axis_values, absolute(ecg_fft))
        # axes[2].title.set_text(s="ECG Magnitude")
        # axes[2].set_ylabel(ylabel="mV")
        # axes[3].plot(x_axis_values, angle(ecg_fft))
        # axes[3].title.set_text(s="ECG Phase")
        # axes[3].set_ylabel(ylabel="radians")
        # axes[3].set_xlabel(xlabel="Hz")
        # show()

        # from matplotlib.pyplot import close, colorbar, figure, imshow, subplot, suptitle, xlabel, xticks, ylabel, yticks
        # from constants import SPECTROGRAM_FREQUENCIES, SPECTROGRAM_SAMPLES
        # _, _, ecg_spectrogram = spectrogram(
        #     x=segment[:, CHANNEL_INDEX_FOR_ECG],
        #     fs=SIGNALS_SAMPLING_FREQUENCY,
        #     nperseg=SPECTROGRAM_SAMPLES_FOR_EACH_SFTF,
        #     noverlap=SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF
        #     # NOTE: this spectrogram is based on real values of power
        #     # spectral density and with no redundant information repeated
        #     # on negative frequencies
        # )
        # _, _, ppg_spectrogram = spectrogram(
        #     x=segment[:, CHANNEL_INDEX_FOR_PPG],
        #     fs=SIGNALS_SAMPLING_FREQUENCY,
        #     nperseg=SPECTROGRAM_SAMPLES_FOR_EACH_SFTF,
        #     noverlap=SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF
        #     # NOTE: this spectrogram is based on real values of power
        #     # spectral density and with no redundant information repeated
        #     # on negative frequencies
        # )
        # x_t = range(len(SPECTROGRAM_SAMPLES))[::4]
        # x_f = range(len(SPECTROGRAM_FREQUENCIES))[::4]
        # figure()
        # suptitle("Time-Frequency-Domain Input Representation")
        # subplot(1, 2, 1)
        # title("PPG Spectrogram")
        # imshow(ppg_spectrogram)
        # colorbar()
        # xlabel("seconds")
        # xticks(ticks=x_t, labels=map(lambda x: '%.1f'%x, SPECTROGRAM_SAMPLES[::4].tolist()))
        # ylabel("Hz")
        # yticks(ticks=x_f, labels=map(lambda x: '%.1f'%x, SPECTROGRAM_FREQUENCIES[::4].tolist()))
        # subplot(1, 2, 2)
        # title("ECG Spectrogram")
        # imshow(ecg_spectrogram)
        # colorbar()
        # xlabel("seconds")
        # xticks(ticks=x_t, labels=map(lambda x: '%.1f'%x, SPECTROGRAM_SAMPLES[::4].tolist()))
        # ylabel("Hz")
        # yticks(ticks=x_f, labels=map(lambda x: '%.1f'%x, SPECTROGRAM_FREQUENCIES[::4].tolist()))
        # subplots_adjust(wspace=0.5)
        # show()
        # close()

        # raise Exception("Plots done.")
        # ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

        # if the signals are to be represented in the frequency domain:
        if INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY:
            abp = segment[:, CHANNEL_INDEX_FOR_ABP]
            ecg_fft = fft(x=segment[:, CHANNEL_INDEX_FOR_ECG])
            ppg_fft = fft(x=segment[:, CHANNEL_INDEX_FOR_PPG])
            segment = empty(shape=(segment.shape[0], 5), dtype=segment.dtype)
            segment[:, CHANNEL_INDEX_FOR_ABP] = abp
            segment[:, CHANNEL_INDEX_FOR_ECG_MAGNITUDE] = absolute(ecg_fft)
            segment[:, CHANNEL_INDEX_FOR_ECG_PHASE] = angle(ecg_fft)
            segment[:, CHANNEL_INDEX_FOR_PPG_MAGNITUDE] = absolute(ppg_fft)
            segment[:, CHANNEL_INDEX_FOR_PPG_PHASE] = angle(ppg_fft)
        # if the signals are to be represented in the time-frequency domain:
        elif INPUT_DOMAIN == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY:
            _, _, ecg_spectrogram = spectrogram(
                x=segment[:, CHANNEL_INDEX_FOR_ECG],
                fs=SIGNALS_SAMPLING_FREQUENCY,
                nperseg=SPECTROGRAM_SAMPLES_FOR_EACH_SFTF,
                noverlap=SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF
                # NOTE: this spectrogram is based on real values of power
                # spectral density and with no redundant information repeated
                # on negative frequencies
            )
            _, _, ppg_spectrogram = spectrogram(
                x=segment[:, CHANNEL_INDEX_FOR_PPG],
                fs=SIGNALS_SAMPLING_FREQUENCY,
                nperseg=SPECTROGRAM_SAMPLES_FOR_EACH_SFTF,
                noverlap=SPECTROGRAM_SAMPLES_OVERLAPPING_OVER_SFTF
                # NOTE: this spectrogram is based on real values of power
                # spectral density and with no redundant information repeated
                # on negative frequencies
            )
            # padding and flattening the spectrograms to save them as signals
            # of the same length of the pressure signal segment for easy
            # reconstruction when loading samples:
            segment[:, CHANNEL_INDEX_FOR_PPG] = pad(
                array=ppg_spectrogram.flatten(order='C'),
                # no padding at the beginning but only at the end:
                pad_width=((0, N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES),),
                mode='edge'
            )
            segment[:, CHANNEL_INDEX_FOR_ECG] = pad(
                array=ppg_spectrogram.flatten(order='C'),
                # no padding at the beginning but only at the end:
                pad_width=((0, N_SPECTROGRAM_FLATTENING_TRAILING_ZEROES),),
                mode='edge'
            )
        # otherwise, the signal is already represented in the time domain

        # optionally saving the signal block segment:
        if save:
            if base_file_name is None:
                raise TypeError("'file_name' argument cannot be 'None'")

            # assigning the signal block segment randomly, but with relative
            # probability (mass) distribution weights proportional to the
            # respective numbers of expected assignemts, to one among the
            # training, the validation and the test set:
            assigned_set_dir = choices(
                population=[
                    UCI_TRAINING_SET_DIR,
                    UCI_VALIDATION_SET_DIR,
                    UCI_TEST_SET_DIR
                ],
                weights=[
                    PORTION_OF_DATA_FOR_TRAINING,
                    PORTION_OF_DATA_FOR_VALIDATION,
                    PORTION_OF_DATA_FOR_TEST
                ]
            )[0]

            segment_file_path = path_join(
                assigned_set_dir,
                (
                    f'{base_file_name}_with_'
                    f'{str(segment_starting_sample_index)}_bos.bin'
                )
            )

            segment.astype('float32').tofile(segment_file_path)

        n_good_segments += 1

    return n_good_segments


if __name__ == '__main__':
    random_seed(0)

    # ------------------------------------------------------------------------
    # loading data while inspecting their statistics:

    # initializing signals' statistics:
    cumulative_signal_length = 0
    maximum_signal_length = None
    minimum_signal_length = None
    n_signal_blocks = 0
    n_signal_block_segments = 0
    signal_blocks_lenghts_for_histogram = []
    physiological_values_histograms = {
        'abp': {
            'edges': arange(start=30, stop=200, step=10),  # [mmHg]
            'counts': None
        },
        'ecg': {
            'edges': arange(start=-1, stop=+2, step=0.1),  # [mV]
            'counts': None
        },
        'ppg': {
            'edges': arange(start=0, stop=5, step=0.1),  # [adimensional]
            'counts': None
        }
    }

    signal_blocks = []

    if PRELIMINARY_INSPECTION:
        inspection_figure, inspection_axes = subplots(3, sharex=True)

    # for each potential file containing dataset samples:
    for filename in listdir(ORIGINAL_UCI_DATASET_DIR):
        # skipping files not belonging to the original UCI dataset:
        if filename.rsplit(sep='.', maxsplit=1)[1] != 'mat':
            print(f"... skipped file {filename}")
            continue

        else:
            print(f"... processing file {filename}")

        file_path = path_join(ORIGINAL_UCI_DATASET_DIR, filename)

        # --------------------------------------------------------------------
        # NOTE: THIS IS JUST AN ALTERNATIVE TO READ THE SAME DAYASET BUT WHEN
        # DOWNLOADED FROM KAGGLE
        # --------------------------------------------------------------------
        # from scipy.io import loadmat
        # all = loadmat(file_path)['p'][0]
        # for single in all:
        #     if True:
        #         current_signal_block = single.transpose()
        #         assert len(current_signal_block.shape) == 2 and current_signal_block.shape[1] == 3
        # --------------------------------------------------------------------

        # from the dataset file:
        with File(name=file_path, mode='r') as file:
            keys = [key for key in file.keys()]

            if len(keys) != 2:
                raise RuntimeError("Unexpected number of keys.")

            keys.remove('#refs#')

            # for each 3D block of signals belonging to the same recording:
            for table_columns in file[keys[0]]:
                if len(table_columns) != 1:
                    raise RuntimeError("Unexpected number of variables.")

                current_signal_block = file[table_columns[0]][:]

                # skipping signal blocks with non-numerical values:
                if isnan(current_signal_block).any():
                    continue

                n_signal_blocks += 1

                # skipping signals not accepted after manual inspection:
                if not PRELIMINARY_INSPECTION:
                    pass  # TODO: decide what to do
                    # --------------------------------------------------------
                    # if not SIGNALS_INDEX_TO_ACCEPTANCE[n_signal_blocks]:
                    #     continue
                    # --------------------------------------------------------

                # updating signals' statistics:
                current_length = current_signal_block.shape[0]
                cumulative_signal_length += current_length
                signal_blocks_lenghts_for_histogram.append(current_length)
                current_block_histograms_counts = {
                    'abp': histogram(
                        a=current_signal_block[:, CHANNEL_INDEX_FOR_ABP],
                        bins=physiological_values_histograms['abp']['edges']
                    )[0],
                    'ecg': histogram(
                        a=current_signal_block[:, CHANNEL_INDEX_FOR_ECG],
                        bins=physiological_values_histograms['ecg']['edges']
                    )[0],
                    'ppg': histogram(
                        a=current_signal_block[:, CHANNEL_INDEX_FOR_PPG],
                        bins=physiological_values_histograms['ppg']['edges']
                    )[0]
                }
                if n_signal_blocks == 1:
                    maximum_signal_length = current_length
                    minimum_signal_length = current_length
                    physiological_values_histograms['abp']['counts'] = (
                        current_block_histograms_counts['abp']
                    )
                    physiological_values_histograms['ecg']['counts'] = (
                        current_block_histograms_counts['ecg']
                    )
                    physiological_values_histograms['ppg']['counts'] = (
                        current_block_histograms_counts['ppg']
                    )
                else:
                    if current_length > maximum_signal_length:
                        maximum_signal_length = current_length
                    if current_length < minimum_signal_length:
                        minimum_signal_length = current_length
                    physiological_values_histograms['abp']['counts'] += (
                        current_block_histograms_counts['abp']
                    )
                    physiological_values_histograms['ecg']['counts'] += (
                        current_block_histograms_counts['ecg']
                    )
                    physiological_values_histograms['ppg']['counts'] += (
                        current_block_histograms_counts['ppg']
                    )

                signal_blocks.append(current_signal_block)

                if PRELIMINARY_INSPECTION:
                    # if not already inspected:
                    if n_signal_blocks not in SIGNALS_INDEX_TO_ACCEPTANCE:
                        x_axis_values = list(
                            range(current_signal_block.shape[0])
                        )

                        block_length = current_signal_block.shape[0]
                        window_size = 1000
                        window_moving_step = 500

                        for window_beginning in range(
                            0,
                            block_length - window_size,
                            window_moving_step
                        ):
                            window_x_axis_values = x_axis_values[
                                window_beginning:(
                                    window_beginning + window_size
                                )
                            ]
                            block_window = current_signal_block[
                                window_beginning:(
                                    window_beginning + window_size
                                ),
                                :
                            ]

                            inspection_axes[0].clear()
                            inspection_axes[1].clear()
                            inspection_axes[2].clear()
                            inspection_figure.suptitle(
                                t="Signal Block " + str(n_signal_blocks)
                            )
                            inspection_axes[0].plot(
                                window_x_axis_values,
                                block_window[:, CHANNEL_INDEX_FOR_PPG]
                            )
                            inspection_axes[0].title.set_text(s="PPG")
                            inspection_axes[1].plot(
                                window_x_axis_values,
                                block_window[:, CHANNEL_INDEX_FOR_ECG]
                            )
                            inspection_axes[1].title.set_text(
                                s="ECG (II Lead)"
                            )
                            inspection_axes[2].plot(
                                window_x_axis_values,
                                block_window[:, CHANNEL_INDEX_FOR_ABP]
                            )
                            inspection_axes[2].title.set_text(s="ABP")
                            show(block=False)
                            pause(interval=0.0000001)

                        is_signal_rejected = True if (
                            input("Reject? [y/n (or anything else)] |: ")
                            == 'y'
                        ) else False

                        with open(SIGNAL_EXAM_RESULTS_PATH, 'a') as yaml_file:
                            yaml_file.write(
                                yaml_dump(
                                    {
                                        n_signal_blocks: False if
                                        is_signal_rejected else True
                                    }
                                )
                            )

                    continue

                # segmenting the signal block while preprocessing the
                # resulting segments and saving them to the file system:
                n_signal_block_segments += (
                    segment_signal_block_and_filter_segments(
                        whole_length_signal_block=current_signal_block,
                        segment_length_in_samples=N_UCI_SIGNAL_SEGMENT_SAMPLES,
                        standardize_inputs_segmentwise=False,
                        save=False,  # TODO: True,
                        base_file_name=str(n_signal_blocks)
                    )
                )

    if PRELIMINARY_INSPECTION:
        quit("Inspection concluded.")

    average_signal_length = round(cumulative_signal_length / n_signal_blocks)

    # ------------------------------------------------------------------------
    # printing signal blocks' statistics:

    print(f"\t{n_signal_blocks} 3D signal blocks extracted:")
    print(f"\t\t-> average length [samples]: {average_signal_length}")
    print(f"\t\t-> maximum length [samples]: {maximum_signal_length}")
    print(f"\t\t-> minimum length [samples]: {minimum_signal_length}")
    print(
        f"\t{n_signal_block_segments} 3D signal block segments with length of "
        f"{N_UCI_SIGNAL_SEGMENT_SAMPLES} samples preprocessed and saved"
    )

    # # ------------------------------------------------------------------------
    # # plotting the signal blocks' lengths histogram:

    # histogram_bins = list(
    #     range(0, maximum_signal_length, 1000)
    # )
    # figure()
    # hist(
    #     x=signal_blocks_lenghts_for_histogram,
    #     bins=histogram_bins,
    #     align='mid',
    #     color='chartreuse',
    #     rwidth=0.8
    # )
    # title(label="Histogram of 3D Signal Blocks' Length [samples]")
    # xticks(
    #     ticks=histogram_bins,
    #     fontsize=6,
    #     rotation=90
    # )
    # show()


    # # ------------------------------------------------------------------------
    # # plotting the distributions of signals' values:

    # plot_blocking_histogram(
    #     bin_counts=physiological_values_histograms['ppg']['counts'],
    #     bin_edges=physiological_values_histograms['ppg']['edges'],
    #     color='mediumslateblue',
    #     title_text="Histogram of PPG Values [adimensional]"
    # )

    # plot_blocking_histogram(
    #     bin_counts=physiological_values_histograms['abp']['counts'],
    #     bin_edges=physiological_values_histograms['abp']['edges'],
    #     color='skyblue',
    #     title_text="Histogram of ABP Values [mmHg]"
    # )

    # plot_blocking_histogram(
    #     bin_counts=physiological_values_histograms['ecg']['counts'],
    #     bin_edges=physiological_values_histograms['ecg']['edges'],
    #     color='violet',
    #     title_text="Histogram of ECG (II Lead) Values [mV]"
    # )

    # ------------------------------------------------------------------------
    # plotting some random signal blocks' segments:
    if False:  # TODO: restore
        plot_random_signal_block_parts_forever(
            whole_signal_blocks=signal_blocks,
            n_blocks=n_signal_blocks,
            minimum_signal_length_ensured=minimum_signal_length
        )

    # ------------------------------------------------------------------------
