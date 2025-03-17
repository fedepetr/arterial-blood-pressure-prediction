"""
Script to preprocess our in-house dataset.
"""
from argparse import ArgumentParser
import sys
import os
from os import getcwd, listdir, remove
from os.path import isdir, join as path_join
from random import choices
from sys import path
from typing import Dict, List, Literal, Tuple, Union, get_args

import json
 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import numpy as np 
from numpy import array, float32 as np_float32, insert, ones, linspace, where
from numpy.typing import NDArray
from openpyxl import load_workbook
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation
from scipy.signal import iirfilter, iirnotch, filtfilt
# pylint: disable=import-error,no-name-in-module
from tensorflow import float32 as tf_float32
from tensorflow.data import TFRecordDataset
from tensorflow.io import FixedLenFeature, parse_single_example, TFRecordWriter
from tensorflow.train import Example, Feature, FloatList, Features
# pylint: enable=import-error,no-name-in-module
from yaml import safe_load as yaml_safe_load

path.append(path_join(getcwd(), 'modeling'))
from constants import (
    IN_HOUSE_DATASET_ANNOTATIONS_DIR,
    IN_HOUSE_DATASET_PARTITIONING_CONFIGURATIONS_DIR,
    IN_HOUSE_DATASET_STATISTICS_DIR,
    IN_HOUSE_TEST_SET_DIR,
    IN_HOUSE_TRAINING_SET_DIR,
    IN_HOUSE_VALIDATION_SET_DIR,
    N_SAMPLES_PER_IN_HOUSE_SEGMENT,
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
    NAME_FOR_IN_HOUSE_PPG_RED,
    NAME_FOR_IN_HOUSE_PPG_INFRARED,
    NAME_FOR_IN_HOUSE_PPG_GREEN,
    NAME_FOR_IN_HOUSE_RESP_PT_1,
    NAME_FOR_IN_HOUSE_RESP_PT_2,
    NAME_FOR_IN_HOUSE_RESP_PT_3,
    NAME_FOR_IN_HOUSE_SCG,
    NAME_FOR_IN_HOUSE_TEMP_BODY,
    NAME_FOR_IN_HOUSE_TEMP_EXT,
    N_IN_HOUSE_INPUT_CHANNELS,
    ORIGINAL_IN_HOUSE_DATASET_DIR,
    PORTION_OF_DATA_FOR_TRAINING,
    PORTION_OF_DATA_FOR_VALIDATION,
    PORTION_OF_DATA_FOR_TEST,
    SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ,
    SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ,
)

# Detect the platform running script to set-up figure plot
from sys import platform
FLAG = 0 
if platform == "darwin":
    FLAG = 0
elif platform == "win32":
    FLAG = 1


FILENAME_FOR_ABP_MASK_SIGNAL = '0B_FISIOCAL.txt'
FILENAME_FOR_ABP_SIGNALS = '0A_BLOODPRESSURE.txt'
FILENAME_FOR_ACC_SIGNALS = '03_ACC.txt'
FILENAME_FOR_ACOUSTIC_SIGNAL = '01_ACOUSTIC.txt'
FILENAME_FOR_BIOIMPEDANCE_SIGNAL = '05_BIOZ.txt'
FILENAME_FOR_ECG_SIGNAL = '00_ECG.txt'
FILENAME_FOR_PPG_SIGNALS = '02_PPG.txt'
FILENAME_FOR_SCG_SIGNAL = '04_SCG.txt'
FILENAME_FOR_TEMP_BODY_SIGNAL = '06_BODYSURFTEMP.txt'
FILENAME_FOR_TEMP_EXT_SIGNAL = '07_AMBIENTTEMP.txt'
FILENAME_FOR_ALL_SUBJECTS_METADATA = 'Demography_Soundi2Finometer.xlsx'
(
    INDEX_FOR_ABP_MASK,
    INDEX_FOR_FINGER_ABP,
    INDEX_FOR_BRACHIAL_ABP
) = list(range(3))
(
    INDEX_FOR_ACC_X,
    INDEX_FOR_ACC_Y,
    INDEX_FOR_ACC_Z,
    INDEX_FOR_ACC_XYZ_MAGN,
    INDEX_FOR_ACOUSTIC,
    INDEX_FOR_BIOIMPEDANCE,
    INDEX_FOR_ECG,
    INDEX_FOR_PPG_RED,
    INDEX_FOR_PPG_INFRARED,
    INDEX_FOR_PPG_GREEN,
    INDEX_FOR_SCG,
    INDEX_FOR_TEMP_BODY,
    INDEX_FOR_TEMP_EXT,
    # NOTE: the following ones must follow, being injected later on:
    INDEX_FOR_RESP_PT_1,
    INDEX_FOR_RESP_PT_2,
    INDEX_FOR_RESP_PT_3
) = list(range(N_IN_HOUSE_INPUT_CHANNELS))
INSPECTION = False
INVALID_FINAL_WINDOW_OF_NON_ABP_SIGNALS_IN_S = 60
INVALID_INITIAL_WINDOW_OF_NON_ABP_SIGNALS_IN_S = 120
MINUTES_AT_REST_AS_INTERVALS = [
    (0, 5),
    (15, 20),
    (25, 30)
]
NON_SUBJECT_DIR_NAMES = [
    '.DS_Store',
    '.gitkeep',
    'desktop.ini',
    'Soundi_Finometer_Datafile.docx',
    'Soundi_Finometer_Protocol.docx',
    FILENAME_FOR_ALL_SUBJECTS_METADATA,
]
PADDING_AROUND_CALIBRATION_WINDOW_OF_ABP_SIGNALS_IN_S = 1
ValidArgumentsForPartitionModalityNames = Literal[
    'record_wise',
    'segment_wise',
    'subject_wise'
]
(
    PARTITION_MODALITY_NAME_FOR_RECORD_WISE,
    PARTITION_MODALITY_NAME_FOR_SEGMENT_WISE,
    PARTITION_MODALITY_NAME_FOR_SUBJECT_WISE
) = get_args(tp=ValidArgumentsForPartitionModalityNames)
ValidArgumentsForSetNames = Literal[
    'training',
    'validation',
    'test'
]
(
    PARTITIONING_SET_NAME_FOR_TRAINING,
    PARTITIONING_SET_NAME_FOR_VALIDATION,
    PARTITIONING_SET_NAME_FOR_TEST
) = get_args(tp=ValidArgumentsForSetNames)
PARTITIONING_CONFIGURATIONS_TYPE = Union[
    # when record-wise:
    Dict[ValidArgumentsForSetNames, List[str]],
    # when segment-wise:
    None,
    # when subject-wise:
    Dict[ValidArgumentsForSetNames, Dict[str, List[str]]]
]

def align_signal_banks_and_remove_invalid_trailing_segments(
        abp_signal_bank: NDArray,
        non_abp_signal_bank: NDArray,
        abp_shifts_in_s: Dict[str, float]
) -> Tuple[Tuple[NDArray, NDArray], int]:
    """
    Aligning the input signal banks and removing the invalid trailing
    segments.
    """
    n_non_abp_samples_cut_at_signal_beginning = 0

    # erasing the initial time window of the non-ABP-related signals during
    # which the ABP-related signals were not recorded:

    non_abp_signal_bank, n_samples_cut = cut_outmost_signals_seconds(
        signal_batch=non_abp_signal_bank,
        initial_seconds=True,
        shift_in_seconds=abp_shifts_in_s['absolute'],
        sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )

    n_non_abp_samples_cut_at_signal_beginning += n_samples_cut

    # aligning the non-ABP-related signals to the ABP-related signals by
    # erasing the initial gap, that is the additional time window of the
    # signals whose bank recording started earlier in time:

    if abp_shifts_in_s['relative'] >= 0:
        abp_signal_bank, _ = cut_outmost_signals_seconds(
            signal_batch=abp_signal_bank,
            initial_seconds=True,
            shift_in_seconds=abp_shifts_in_s['relative'],
            sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
        )

    else:
        non_abp_signal_bank, n_samples_cut = cut_outmost_signals_seconds(
            signal_batch=non_abp_signal_bank,
            initial_seconds=True,
            shift_in_seconds=(-1 * abp_shifts_in_s['relative']),
            sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
        )

        n_non_abp_samples_cut_at_signal_beginning += n_samples_cut

    # removing the invalid, initial calibration window of all signals, which
    # are now aligned:

    abp_signal_bank, _ = cut_outmost_signals_seconds(
        signal_batch=abp_signal_bank,
        initial_seconds=True,
        shift_in_seconds=INVALID_INITIAL_WINDOW_OF_NON_ABP_SIGNALS_IN_S,
        sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
    )

    non_abp_signal_bank, n_samples_cut = cut_outmost_signals_seconds(
        signal_batch=non_abp_signal_bank,
        initial_seconds=True,
        shift_in_seconds=INVALID_INITIAL_WINDOW_OF_NON_ABP_SIGNALS_IN_S,
        sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )

    n_non_abp_samples_cut_at_signal_beginning += n_samples_cut

    # removing the invalid, final calibration window of all signals, which are
    # now aligned:

    abp_signal_bank, _ = cut_outmost_signals_seconds(
        signal_batch=abp_signal_bank,
        initial_seconds=False,
        shift_in_seconds=INVALID_FINAL_WINDOW_OF_NON_ABP_SIGNALS_IN_S,
        sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
    )

    non_abp_signal_bank, _ = cut_outmost_signals_seconds(
        signal_batch=non_abp_signal_bank,
        initial_seconds=False,
        shift_in_seconds=INVALID_FINAL_WINDOW_OF_NON_ABP_SIGNALS_IN_S,
        sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )

    # discaring the final parts with possible artifacts of the non-ABP-related
    # signal bank:

    # TODO
    # import pdb; pdb.set_trace()

    # discarding the exceeding signal portions of the bank whose signals are
    # longer:

    exceeding_seconds_of_abp_over_non_abp_signals = (
        (
            abp_signal_bank.shape[1] /
            SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
        ) - (
            non_abp_signal_bank.shape[1] /
            SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
        )
    )

    if exceeding_seconds_of_abp_over_non_abp_signals >= 0:
        abp_signal_bank, _ = cut_outmost_signals_seconds(
            signal_batch=abp_signal_bank,
            initial_seconds=False,
            shift_in_seconds=exceeding_seconds_of_abp_over_non_abp_signals,
            sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
        )

    else:
        non_abp_signal_bank, _ = cut_outmost_signals_seconds(
            signal_batch=non_abp_signal_bank,
            initial_seconds=False,
            shift_in_seconds=(-1 * exceeding_seconds_of_abp_over_non_abp_signals),
            sampling_frequency=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
        )

    return (
        (abp_signal_bank, non_abp_signal_bank),
        n_non_abp_samples_cut_at_signal_beginning
    )


def check_directory_contains_expected_signal_files(dir_: str) -> bool:
    """
    Check that the filenames in the given directory match the expected ones.
    """
    actual_files = listdir(dir_)
    if '.DS_Store' in actual_files:
        actual_files.remove('.DS_Store')
    if 'desktop.ini' in actual_files:
        actual_files.remove('desktop.ini')
    return (
        set(actual_files) == {
            FILENAME_FOR_ABP_MASK_SIGNAL,
            FILENAME_FOR_ABP_SIGNALS,
            FILENAME_FOR_ACC_SIGNALS,
            FILENAME_FOR_ACOUSTIC_SIGNAL,
            FILENAME_FOR_BIOIMPEDANCE_SIGNAL,
            FILENAME_FOR_ECG_SIGNAL,
            FILENAME_FOR_PPG_SIGNALS,
            FILENAME_FOR_SCG_SIGNAL,
            FILENAME_FOR_TEMP_BODY_SIGNAL,
            FILENAME_FOR_TEMP_EXT_SIGNAL,
        }
    )


def clear_sets_of_tf_records() -> None:
    """
    Delete the TF records saved in the training, validation and test set
    folders.
    """
    for dir_ in (
            IN_HOUSE_TEST_SET_DIR,
            IN_HOUSE_TRAINING_SET_DIR,
            IN_HOUSE_VALIDATION_SET_DIR
    ):
        tf_records_filenames = listdir(dir_)
        tf_records_filenames.remove('.gitkeep')
        for filename in tf_records_filenames:
            remove(path=path_join(dir_, filename))


def compose_record_id(subject_name: str, record_date: str) -> str:
    """
    Compose the subject name and the record date so as to create the record ID.
    """
    return f'{subject_name}-{record_date}'


def create_tfrecord_example_from_signals_segments(
        abp_segments: NDArray,
        non_abp_segments: NDArray,
        moving: bool

) -> Example:
    """
    Create a TFRecord example containing a feature, made of a 1-D signal, for
    each signal segment.
    """
    return Example(
        features=Features(
            feature={
                #Meta Data
                NAME_FOR_IN_HOUSE_METADATA: Feature(
                    float_list=FloatList(
                        value= [moving]
                    )
                ),

                # ABP-related signals:

                NAME_FOR_IN_HOUSE_ABP_MASK: Feature(
                    float_list=FloatList(
                        value=abp_segments[INDEX_FOR_ABP_MASK]
                    )
                ),
                NAME_FOR_IN_HOUSE_FINGER_ABP: Feature(
                    float_list=FloatList(
                        value=abp_segments[
                            INDEX_FOR_FINGER_ABP
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_BRACHIAL_ABP: Feature(
                    float_list=FloatList(
                        value=abp_segments[
                            INDEX_FOR_BRACHIAL_ABP
                        ]
                    )
                ),

                # non-ABP-related signals:

                NAME_FOR_IN_HOUSE_ACC_XYZ_MAGN: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_ACC_XYZ_MAGN
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_ACC_X: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_ACC_X
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_ACC_Y: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_ACC_Y
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_ACC_Z: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_ACC_Z
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_ACOUSTIC: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_ACOUSTIC
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_BIOIMPEDANCE: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_BIOIMPEDANCE
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_ECG: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_ECG
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_PPG_GREEN: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_PPG_GREEN
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_PPG_INFRARED: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_PPG_INFRARED
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_PPG_RED: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_PPG_RED
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_RESP_PT_1: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_RESP_PT_1
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_RESP_PT_2: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_RESP_PT_2
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_RESP_PT_3: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_RESP_PT_3
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_SCG: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_SCG
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_TEMP_BODY: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_TEMP_BODY
                        ]
                    )
                ),
                NAME_FOR_IN_HOUSE_TEMP_EXT: Feature(
                    float_list=FloatList(
                        value=non_abp_segments[
                            INDEX_FOR_TEMP_EXT
                        ]
                    )
                ),
            }
        )
    )


def cut_outmost_signals_seconds(
        signal_batch: NDArray,
        initial_seconds: bool,
        shift_in_seconds: float,
        sampling_frequency: float
) -> NDArray:
    """
    Cutting the initial/final signals' samples (as if shifting them leftwards/
    rightwards) so as to satisfy the desired time cut (time shift), given the
    signals' sampling frequency.
    """
    if shift_in_seconds < 0:
        raise Exception("Ill-conceived code.")

    shift_in_samples = round(shift_in_seconds * sampling_frequency)

    # if the initial seconds are to be discarded:
    if initial_seconds:
        truncated_signal_batch = signal_batch[:, shift_in_samples:]
    # if the final seconds are to be discarded:
    else:
        truncated_signal_batch = signal_batch[:, :-shift_in_samples]
    
    return truncated_signal_batch, shift_in_samples


def downsample_abp_related_and_segment_and_save_segments_to_sets(
        abp_signal_bank: NDArray,
        non_abp_signal_bank: NDArray,
        resting_periods: List[Tuple[int, int]],
        resting_periods_alignment_offset: int,
        subject: str,
        record_date: str,
        annotation_mode: bool,
        partitioning_modality: ValidArgumentsForPartitionModalityNames,
        partitioning_configurations: PARTITIONING_CONFIGURATIONS_TYPE
) -> None:
    """
    Downsample the ABP-related signals, split all signals in equal-length
    segments and save such segments as TF records randomly assigning them to
    either the training, the validation or the test set locations.
    """
    final_n_samples_per_signal = non_abp_signal_bank.shape[1]

    # adjusting the resting periods' sample indexes to the aligned signals:
    resting_periods = [
        tuple(
            map(
                lambda index: index - resting_periods_alignment_offset,
                extremes_sample_indexes
            )
        ) for extremes_sample_indexes in resting_periods
    ]

    abp_signal_bank = downsample_abp_related_signals(
        signals=abp_signal_bank,
        new_n_equispaced_samples=final_n_samples_per_signal
    )

    record_id = compose_record_id(
        subject_name=subject,
        record_date=record_date
    )

    statistics_file_path = path_join(
        IN_HOUSE_DATASET_STATISTICS_DIR,
        f'{record_id}.csv'
    )

    training_tfrecord_path = path_join(
        IN_HOUSE_TRAINING_SET_DIR,
        record_id + '.tfrecord'
    )
    validation_tfrecord_path = path_join(
        IN_HOUSE_VALIDATION_SET_DIR,
        record_id + '.tfrecord'
    )
    test_tfrecord_path = path_join(
        IN_HOUSE_TEST_SET_DIR,
        record_id + '.tfrecord'
    )

    with (
            open(statistics_file_path, 'w') as statistics_file,
            TFRecordWriter(path=training_tfrecord_path) as (
                training_tfrecord_writer
            ),
            TFRecordWriter(path=validation_tfrecord_path) as (
                validation_tfrecord_writer
            ),
            TFRecordWriter(path=test_tfrecord_path) as (
                test_tfrecord_writer
            )
    ):
        initial_sample_index = 0
        final_sample_index = N_SAMPLES_PER_IN_HOUSE_SEGMENT
        
        while final_sample_index <= final_n_samples_per_signal:
            # # filtering segments recorded not at rest out:
            # to_be_skipped = True
            # for leftmost_index, rightmost_index in resting_periods:
            #     if (
            #             leftmost_index <= initial_sample_index
            #             and final_sample_index <= rightmost_index
            #     ):
            #         to_be_skipped = False
            #         break

            # controll if we are in rest/movement 
            is_moving = True
            for leftmost_index, rightmost_index in resting_periods:
                if (
                        leftmost_index <= initial_sample_index
                        and final_sample_index <= rightmost_index
                ):
                    is_moving = False
                    break


            # if not to_be_skipped:
            if True:
                abp_signal_bank_segment = abp_signal_bank[
                    :,
                    initial_sample_index:final_sample_index
                ]
                non_abp_signal_bank_segment = non_abp_signal_bank[
                    :,
                    initial_sample_index:final_sample_index
                ]
            

                if annotation_mode:
                    # check if the record exist, if true load up to the last saved point 
                    # otherwise create the file and start with the analysis

                    if os.path.exists(path_join(IN_HOUSE_DATASET_ANNOTATIONS_DIR , record_id + '.json' )):
                        
                        with open(path_join(IN_HOUSE_DATASET_ANNOTATIONS_DIR, record_id + '.json'), "r")as file:
                            data = json.load(file)
                            chiavi = list(data.keys())
                        
                        if (f'{initial_sample_index}-{final_sample_index}' in chiavi ) == True :
                            print('LOAD... Chunk analyzed ---> initial index: '+str(initial_sample_index)+'| final index: '+ str(final_sample_index))
                            
                            initial_sample_index += N_SAMPLES_PER_IN_HOUSE_SEGMENT
                            final_sample_index += N_SAMPLES_PER_IN_HOUSE_SEGMENT

                            continue

                    else:
                        data = {}
                        with open(path_join(IN_HOUSE_DATASET_ANNOTATIONS_DIR, record_id + '.json'), "w")as file:
                             json.dump(obj = data, fp = file)

                    # creating plot

                    fig, ax = plt.subplots(7,1)
                    fig.tight_layout()
                    fig.set_figheight(8)
                    fig.set_figwidth(8)

                    time = np.linspace(initial_sample_index,final_sample_index,final_sample_index - initial_sample_index)
                    ax[0].plot(time, non_abp_signal_bank_segment[INDEX_FOR_ECG])
                    ax[0].set_title('ECG')
                    ax[1].plot(time,non_abp_signal_bank_segment[INDEX_FOR_PPG_GREEN])
                    ax[1].set_title('PPG_GREEN')
                    ax[2].plot(time,non_abp_signal_bank_segment[INDEX_FOR_ACOUSTIC])
                    ax[2].set_title('PCG')
                    ax[3].plot(time,non_abp_signal_bank_segment[INDEX_FOR_SCG])
                    ax[3].set_title('SCG')
                    ax[4].plot(time,non_abp_signal_bank_segment[INDEX_FOR_BIOIMPEDANCE])
                    ax[4].set_title('BIO_IMP')
                    ax[5].plot(time,abp_signal_bank_segment[INDEX_FOR_BRACHIAL_ABP])
                    ax[5].set_title('BRACHIAL_ABP')
                    ax[6].plot(time,abp_signal_bank_segment[INDEX_FOR_ABP_MASK])
                    ax[6].set_title('ABP_MASK')
                    
                    # ax[5][0].plot(non_abp_signal_bank_segment[5,:])


                    #defining fuction to save the selected chunk 
                    def save_chunk(b):

                        #JSON data defined
                        with open(path_join( IN_HOUSE_DATASET_ANNOTATIONS_DIR , record_id +'.json'), "r") as file:
                                data = json.load(file)
                                
                            
                        data[f'{initial_sample_index}-{final_sample_index}']= True
                        
                        with open(path_join(IN_HOUSE_DATASET_ANNOTATIONS_DIR, record_id + '.json'), "w")as file:
                             json.dump(obj = data, fp = file)
                             

                        plt.close()
                        print('Chunk analyzed | V ---> initial index: '+str(initial_sample_index)+'| final index: '+ str(final_sample_index))


                    #defining fuction to discard selected chuk 
                    def delete_chunk(b):

                       #JSON data defined
                        with open(path_join( IN_HOUSE_DATASET_ANNOTATIONS_DIR , record_id +'.json'), "r") as file:
                                data = json.load(file)
                                
                            
                        data[f'{initial_sample_index}-{final_sample_index}']= False
                        
                        with open(path_join(IN_HOUSE_DATASET_ANNOTATIONS_DIR, record_id + '.json'), "w")as file:
                             json.dump(obj = data, fp = file)
                        
                        
                        plt.close()
                        print('Chunk analyzed | X ---> initial index: '+str(initial_sample_index)+'| final index: '+ str(final_sample_index))

                    
                    # defining buttons and add its functionality

                    #save button
                    axsave = fig.add_axes([0.6, 0, 0.1, 0.055])
                    bsave = Button(axsave, 'Save')
                    bsave.on_clicked(save_chunk)

                    #delete 
                    axdelete = fig.add_axes([0.8, 0, 0.1, 0.055])
                    bdelete = Button(axdelete, 'Delete')
                    bdelete.on_clicked(delete_chunk)
                    
                    fig.suptitle(record_id + f'{is_moving}', fontsize=16,)
                    plt.subplots_adjust(top=0.90)
                    
                    figManager = plt.get_current_fig_manager()
                    if FLAG == 0 :
                        figManager.full_screen_toggle()
                    else:
                        figManager.window.showMaximized()

                    plt.show()

                else:
                    #controllo che nella cartella IN_HOUSE_DATASET_ANNOTATIONS sia presente il folder con i record selezionati manualmente altrimenti vado avanti con l'analisi 
                    if os.path.exists(path_join(IN_HOUSE_DATASET_ANNOTATIONS_DIR,record_id + '.json')) :

                        #scorro tutti i record all'interno della cartella e prendo unicamente quelli salvati

                        with open(path_join( IN_HOUSE_DATASET_ANNOTATIONS_DIR , record_id +'.json'), "r") as file:
                                data = json.load(file)
                        try:

                            if data[f'{initial_sample_index}-{final_sample_index}'] :

                                print('Chunk analyzed | V ---> initial index: ' + str(initial_sample_index) + ' | final index: ' + str(final_sample_index))
                                valid_abp_samples = abp_signal_bank_segment[
                                    INDEX_FOR_BRACHIAL_ABP
                                ][
                                    where(abp_signal_bank_segment[INDEX_FOR_ABP_MASK] == False)
                                ]
                                if valid_abp_samples.shape[0]:  # if not empty
                                    statistics_file.write(
                                        f'{valid_abp_samples.min()},' +
                                        f'{valid_abp_samples.max()}\n'
                                    )

                                # creating a TF example corresponding to the current signals'
                                # segment:
                                example = create_tfrecord_example_from_signals_segments(
                                    abp_segments=abp_signal_bank_segment,
                                    non_abp_segments=non_abp_signal_bank_segment,
                                    moving=is_moving
                                )

                                # choosing which set the current segment will
                                # be assigned to:

                                if partitioning_modality == PARTITION_MODALITY_NAME_FOR_RECORD_WISE:
                                    error_message = (
                                        "Ill-defined partitioning for " +
                                        f"record {record_date} of " +
                                        f"subject {subject_id}."
                                    )
                                    try:
                                        chosen_set = partitioning_configurations[subject_id][record_date]
                                        if chosen_set == PARTITIONING_SET_NAME_FOR_TRAINING:
                                            writer = training_tfrecord_writer
                                        elif chosen_set == PARTITIONING_SET_NAME_FOR_VALIDATION:
                                            writer = validation_tfrecord_writer
                                        elif chosen_set == PARTITIONING_SET_NAME_FOR_TEST:
                                            writer = test_tfrecord_writer
                                        else:
                                            raise Exception("Ill-conceived code.")
                                    except Exception as exception:
                                        raise Exception(error_message) from exception

                                elif partitioning_modality == PARTITION_MODALITY_NAME_FOR_SEGMENT_WISE:
                                    # random assignation, weighted by set sizes:
                                    writer = choices(
                                        population=[
                                            training_tfrecord_writer,
                                            validation_tfrecord_writer,
                                            test_tfrecord_writer
                                        ],
                                        weights=[
                                            PORTION_OF_DATA_FOR_TRAINING,
                                            PORTION_OF_DATA_FOR_VALIDATION,
                                            PORTION_OF_DATA_FOR_TEST
                                        ]
                                    )[0]

                                elif partitioning_modality == PARTITION_MODALITY_NAME_FOR_SUBJECT_WISE:
                                    error_message = (
                                        "Ill-defined partitioning for " +
                                        f"subject {subject_id}."
                                    )
                                    try:
                                        chosen_set = partitioning_configurations[subject_id]
                                        if chosen_set == PARTITIONING_SET_NAME_FOR_TRAINING:
                                            writer = training_tfrecord_writer
                                        elif chosen_set == PARTITIONING_SET_NAME_FOR_VALIDATION:
                                            writer = validation_tfrecord_writer
                                        elif chosen_set == PARTITIONING_SET_NAME_FOR_TEST:
                                            writer = test_tfrecord_writer
                                        else:
                                            raise Exception("Ill-conceived code.")
                                    except Exception as exception:
                                        raise Exception(error_message) from exception

                                else:
                                    raise Exception("Ill-conceived code.")

                                # saving the current segment's record to the assigned set:
                                writer.write(record=example.SerializeToString())

                        except KeyError:
                            print('File incomplete, return to "ANNOTATION" mode and complete the analysis.....' )
                            sys.exit()

                    else:
                        print(
                            f"⚠️ Record '{record_date}' of subject " +
                            f"'{subject}' not yet annotated for discarded " +
                            "segments, so discarded completely."
                        )

            initial_sample_index += N_SAMPLES_PER_IN_HOUSE_SEGMENT
            final_sample_index += N_SAMPLES_PER_IN_HOUSE_SEGMENT

    print(
        f"✅ Signal segments of record '{record_date}' of subject '{subject}' "
        "written to disk as TF records."
    )


def downsample_abp_related_signals(
        signals: NDArray,
        new_n_equispaced_samples: int
) -> NDArray:
    """
    Downsample the ABP-related signals as specified.

    NOTE: nearest-neighbor interpolation employed in order to comply with
    signals representing masks.
    """
    old_abscissas = linspace(0, signals.shape[1], signals.shape[1])
    new_abscissas = linspace(0, signals.shape[1], new_n_equispaced_samples)

    interpolator = interp1d(x=old_abscissas, y=signals, kind='nearest', axis=-1)
    return interpolator(x=new_abscissas)


def filter_signal_banks(
        abp_signal_bank: NDArray,
        non_abp_signal_bank: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Filter each signal of the two signal banks properly.
    """
    # NOTE: no filtering applied to ABP mask, being it just a mere calibration
    # mask and not a physiological signal

    # ABP:

    filter_coefficients = iirfilter(
        N=2,
        Wn=[25],
        btype='lowpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
    )
    abp_signal_bank[INDEX_FOR_BRACHIAL_ABP] = filtfilt(
        *filter_coefficients,
        x=abp_signal_bank[INDEX_FOR_BRACHIAL_ABP]
    )
    abp_signal_bank[INDEX_FOR_FINGER_ABP] = filtfilt(
        *filter_coefficients,
        x=abp_signal_bank[INDEX_FOR_FINGER_ABP]
    )

    # ABP MASK:

    abp_signal_bank[INDEX_FOR_ABP_MASK] = binary_dilation(
        input=abp_signal_bank[INDEX_FOR_ABP_MASK],
        structure=ones(3,),
        iterations=(
            PADDING_AROUND_CALIBRATION_WINDOW_OF_ABP_SIGNALS_IN_S *
            SAMPLING_FREQUENCY_OF_IN_HOUSE_ABP_SIGNALS_IN_HZ
        )
    )

    # ACC:

    # NOTE: for the respiration component, see later on
    filter_coefficients = iirfilter(
        N=2,
        Wn=[0.05, 1],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    resp_pt_1_signal = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACC_XYZ_MAGN]
    )

    filter_coefficients = iirfilter(
        N=2,
        Wn=[0.05, 1],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_ACC_X] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACC_X]
    )
    non_abp_signal_bank[INDEX_FOR_ACC_Y] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACC_Y]
    )
    non_abp_signal_bank[INDEX_FOR_ACC_Z] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACC_Z]
    )
    non_abp_signal_bank[INDEX_FOR_ACC_XYZ_MAGN] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACC_XYZ_MAGN]
    )

    # ACOUSTIC:

    # NOTE: for the respiration component, see later on
    filter_coefficients = iirfilter(
        N=2,
        Wn=[0.05, 1],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    resp_pt_2_signal = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACOUSTIC]
    )

    filter_coefficients = iirfilter(
        N=2,
        Wn=[10, 40],
        btype='bandpass',
        ftype='butter',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_ACOUSTIC] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ACOUSTIC]
    )

    # BIOIMPEDANCE:

    # NOTE: for the respiration component, see later on
    filter_coefficients = iirfilter(
        N=2,
        Wn=[0.05, 1],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    resp_pt_3_signal = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_BIOIMPEDANCE]
    )

    filter_coefficients = iirfilter(
        N=2,
        Wn=[1, 5],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_BIOIMPEDANCE] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_BIOIMPEDANCE]
    )

    # ECG:

    filter_coefficients = iirfilter(
        N=2,
        Wn=[5, 25],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_ECG] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ECG]
    )

    filter_coefficients = iirnotch(
        w0=25,
        Q=25/5,
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_ECG] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ECG]
    )

    filter_coefficients = iirnotch(
        w0=36,
        Q=36/5,
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_ECG] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ECG]
    )

    filter_coefficients = iirnotch(
        w0=50,
        Q=50/5,
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_ECG] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ECG]
    )

    # PPG:

    non_abp_signal_bank[INDEX_FOR_PPG_GREEN] = vertically_invert_signal(
        signal=non_abp_signal_bank[INDEX_FOR_PPG_GREEN]
    )
    non_abp_signal_bank[INDEX_FOR_PPG_INFRARED] = vertically_invert_signal(
        signal=non_abp_signal_bank[INDEX_FOR_PPG_INFRARED]
    )
    non_abp_signal_bank[INDEX_FOR_PPG_RED] = vertically_invert_signal(
        signal=non_abp_signal_bank[INDEX_FOR_PPG_RED]
    )

    filter_coefficients = iirfilter(
        N=4,
        Wn=[0.5, 10],
        btype='bandpass',
        ftype='bessel',
        fs=SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )
    non_abp_signal_bank[INDEX_FOR_PPG_GREEN] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_PPG_GREEN]
    )
    non_abp_signal_bank[INDEX_FOR_PPG_INFRARED] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_PPG_INFRARED]
    )
    non_abp_signal_bank[INDEX_FOR_PPG_RED] = filtfilt(
        *filter_coefficients,
        x=non_abp_signal_bank[INDEX_FOR_ECG]
    )

    # RESPIRATION - herein created as a result of preprocessing:

    non_abp_signal_bank = insert(
        arr=non_abp_signal_bank,
        obj=INDEX_FOR_RESP_PT_1,
        values=resp_pt_1_signal,
        axis=0
    )
    non_abp_signal_bank = insert(
        arr=non_abp_signal_bank,
        obj=INDEX_FOR_RESP_PT_2,
        values=resp_pt_2_signal,
        axis=0
    )
    non_abp_signal_bank = insert(
        arr=non_abp_signal_bank,
        obj=INDEX_FOR_RESP_PT_3,
        values=resp_pt_3_signal,
        axis=0
    )

    # NOTE: no filtering applied to SCG signal as raw samples are already
    # filtered

    return (abp_signal_bank, non_abp_signal_bank)


def get_required_arguments() -> Tuple[bool]:
    """
    Parse the Python command arguments to check for and eventually return the
    argument representing whether we are in manual annotation mode or not and
    the argument specifying the name of the file to read for data partitioning
    configurations.
    """
    argument_name_for_annotation_mode_as_automatic = "automatic"
    argument_name_for_annotation_mode_as_manual = "manual"

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        '-m',
        '--mode',
        choices=[
            argument_name_for_annotation_mode_as_automatic,
            argument_name_for_annotation_mode_as_manual
        ],
        required=True,
        type=str,
        dest='mode'
    )

    argument_parser.add_argument(
        '-p',
        '--partitioning',
        required=True,
        type=str,
        dest='partitioning'
    )

    parsed_args = argument_parser.parse_args()

    return (
        parsed_args.mode == argument_name_for_annotation_mode_as_manual,
        parsed_args.partitioning
    )


def vertically_invert_signal(signal: NDArray) -> NDArray:
    """
    Vertically invert the signal between while keeping it in the range between
    its maximum and minimum.
    """
    maximum = signal.max()
    minimum = signal.min()
    return (-signal) + (minimum + maximum)


def minute_to_non_abp_sample_index(minute: float) -> int:
    """
    Turn the input minute into the corresponding sample index for a non-ABP
    signal.
    """
    return round(
        (minute * 60) * SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ
    )


def read_signals(dir: str) -> Tuple[NDArray, NDArray]:
    """
    Load each signal/signal component from the respective file.
    """
    # ABP mask:
    mask_flags = []
    with open(path_join(dir, FILENAME_FOR_ABP_MASK_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            mask_flags.append(bool(float(line)))

    # ABP:
    finger_abp_signal = []
    brachial_abp_signal = []
    with open(path_join(dir, FILENAME_FOR_ABP_SIGNALS), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            timestep_pressures = line.split(' ')
            finger_abp_signal.append(float(timestep_pressures[0]))
            brachial_abp_signal.append(float(timestep_pressures[1]))

    # ACC:
    x_acceleration_signal = []
    y_acceleration_signal = []
    z_acceleration_signal = []
    xyz_acc_magn_signal = []
    with open(path_join(dir, FILENAME_FOR_ACC_SIGNALS), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            timestep_values = line.split(',')
            x_acceleration_signal.append(float(timestep_values[0]))
            y_acceleration_signal.append(float(timestep_values[1]))
            z_acceleration_signal.append(float(timestep_values[2]))
            xyz_acc_magn_signal.append(float(timestep_values[3]))

    # ACOUSTIC:
    acoustic_signal = []
    with open(path_join(dir, FILENAME_FOR_ACOUSTIC_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            acoustic_signal.append(float(line))

    # BIOIMPEDANCE:
    bioimpedance_signal = []
    with open(path_join(dir, FILENAME_FOR_BIOIMPEDANCE_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            bioimpedance_signal.append(float(line))

    # ECG:
    ecg_signal = []
    with open(path_join(dir, FILENAME_FOR_ECG_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            ecg_signal.append(float(line))

    # PPG:
    ppg_green_signal = []
    ppg_infrared_signal = []
    ppg_red_signal = []
    with open(path_join(dir, FILENAME_FOR_PPG_SIGNALS), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            timestep_components = line.split(',')
            ppg_green_signal.append(float(timestep_components[2]))
            ppg_infrared_signal.append(float(timestep_components[1]))
            ppg_red_signal.append(float(timestep_components[0]))

    # SCG:
    scg_signal = []
    with open(path_join(dir, FILENAME_FOR_SCG_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            scg_signal.append(float(line))

    # TEMP:
    temp_body_signal = []
    with open(path_join(dir, FILENAME_FOR_TEMP_BODY_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            temp_body_signal.append(float(line))
    temp_ext_signal = []
    with open(path_join(dir, FILENAME_FOR_TEMP_EXT_SIGNAL), 'r') as file:
        for line in file:
            if line.isspace():
                continue
            temp_ext_signal.append(float(line))

    abp_bank = [None] * 3
    non_abp_bank = [None] * 13

    abp_bank[INDEX_FOR_ABP_MASK] = mask_flags
    abp_bank[INDEX_FOR_FINGER_ABP] = finger_abp_signal
    abp_bank[INDEX_FOR_BRACHIAL_ABP] = brachial_abp_signal

    # NOTE: any other non-ABP related signal is longer than the shortest one
    # among the ABP-related ones, where temperature signals and the ECG singal
    # are candidates for the shortest one, yet they are alredy aligned,
    # meaning that the final, exceeding portion of the other signals can be
    # discarded to have all of them of the same length:
    n_samples = min(
        len(ecg_signal),
        len(temp_body_signal),
        len(temp_ext_signal)
    )
    non_abp_bank[INDEX_FOR_ACC_X] = x_acceleration_signal[:n_samples]
    non_abp_bank[INDEX_FOR_ACC_Y] = y_acceleration_signal[:n_samples]
    non_abp_bank[INDEX_FOR_ACC_Z] = z_acceleration_signal[:n_samples]
    non_abp_bank[INDEX_FOR_ACC_XYZ_MAGN] = xyz_acc_magn_signal[:n_samples]
    non_abp_bank[INDEX_FOR_ACOUSTIC] = acoustic_signal[:n_samples]
    non_abp_bank[INDEX_FOR_BIOIMPEDANCE] = bioimpedance_signal[:n_samples]
    non_abp_bank[INDEX_FOR_ECG] = ecg_signal[:n_samples]
    non_abp_bank[INDEX_FOR_PPG_GREEN] = ppg_green_signal[:n_samples]
    non_abp_bank[INDEX_FOR_PPG_INFRARED] = ppg_infrared_signal[:n_samples]
    non_abp_bank[INDEX_FOR_PPG_RED] = ppg_red_signal[:n_samples]
    non_abp_bank[INDEX_FOR_SCG] = scg_signal[:n_samples]
    non_abp_bank[INDEX_FOR_TEMP_BODY] = temp_body_signal[:n_samples]
    non_abp_bank[INDEX_FOR_TEMP_EXT] = temp_ext_signal[:n_samples]

    return (
        array(object=abp_bank, dtype=np_float32),
        array(object=non_abp_bank, dtype=np_float32)
    )


def read_partitioning_configurations(path: str) -> Tuple[
        ValidArgumentsForPartitionModalityNames,
        PARTITIONING_CONFIGURATIONS_TYPE
]:
    """
    Read the partitioning configurations to split the dataset into training,
    validation and test sets, carrying out the necessary validation and
    parsing of some of the specified options.
    """
    error_message_for_n_enabled_modalities = (
        "One and only one dataset partitioning modality must be enabled."
    )

    with open(path, 'r') as file:
        partitioning_configurations = yaml_safe_load(stream=file)

    chosen_modality = None
    n_enabled_modalities = 0
    for modality in partitioning_configurations.keys():
        assert modality in get_args(tp=ValidArgumentsForPartitionModalityNames)
        if partitioning_configurations[modality]['enabled']:
            n_enabled_modalities += 1
            chosen_modality = modality
    assert n_enabled_modalities == 1, error_message_for_n_enabled_modalities

    return (
        chosen_modality,
        partitioning_configurations[chosen_modality]['partitioning']
    )


def read_records_metadata(dir_: str) -> Dict[
        str, Dict[
            str, Union[
                Dict[str, float],
                List[Tuple[int, int]]
            ]
        ]
]:
    """
    Read metadata such as the relative time shift and the absolute delay of
    the ABP signals and the resting time periods, for each subject record.
    """
    records_metadata = {}
    workbook = load_workbook(
        filename=path_join(dir_, FILENAME_FOR_ALL_SUBJECTS_METADATA)
    )
    is_first_header = True
    is_second_header = False
    for record_metadata in workbook.active.rows:
        if is_first_header:
            is_first_header = False
            is_second_header = True
            continue
        elif is_second_header:
            is_second_header = False
            continue
        subject_id = record_metadata[0].value
        record_date = record_metadata[5].value
        relative_time_shift_in_s = record_metadata[9].value
        absolute_delay_in_s = record_metadata[10].value
        if (
                subject_id is None
                or relative_time_shift_in_s is None
                or absolute_delay_in_s is None
        ):
            continue
        subject_record_id = compose_record_id(
            subject_name=subject_id,
            record_date=record_date.strftime('%y_%m_%d')
        )
        records_metadata[subject_record_id] = {
            'abp_shifts_in_s': {  # in seconds
                'relative': relative_time_shift_in_s,
                'absolute': absolute_delay_in_s
            },
            'resting_periods': [  # as indexes of samples
                tuple(map(minute_to_non_abp_sample_index, minutes))
                for minutes in MINUTES_AT_REST_AS_INTERVALS
            ]
        }
    return records_metadata


if __name__ == '__main__':
    clear_sets_of_tf_records()

    (
        is_annotation_mode,
        partitioning_configurations_filename
    ) = get_required_arguments()

    (
        dataset_partitioning_modality,
        dataset_partitioning_configurations
    ) = read_partitioning_configurations(
        path=path_join(
            IN_HOUSE_DATASET_PARTITIONING_CONFIGURATIONS_DIR,
            partitioning_configurations_filename
        )
    )

    subjects_dir_names = listdir(ORIGINAL_IN_HOUSE_DATASET_DIR)

    records_info = read_records_metadata(dir_=ORIGINAL_IN_HOUSE_DATASET_DIR)

    for subject_id in subjects_dir_names:
        if subject_id in NON_SUBJECT_DIR_NAMES:
            continue

        subject_records_dir = path_join(
            ORIGINAL_IN_HOUSE_DATASET_DIR,
            subject_id
        )

        for record_dir_name in listdir(subject_records_dir):
            if record_dir_name in ('.DS_Store', 'desktop.ini'):
                continue

            record_signals_dir = path_join(
                subject_records_dir,
                record_dir_name,
                'raw'
            )

            if (
                    not isdir(record_signals_dir)
                    or not check_directory_contains_expected_signal_files(
                        dir_=record_signals_dir,
                    )
            ):
                print(
                    f"❌ Discarded record '{record_dir_name}' of subject",
                    f"'{subject_id}' because of missing/unexpected raw",
                    "signal file(s)."
                )
                continue

            print(
                f"... Processing record '{record_dir_name}'",
                f"of subject '{subject_id}'..."
            )
            
            record_id = compose_record_id(
                subject_name=subject_id,
                record_date=record_dir_name
            )

            abp_signals, non_abp_signals = read_signals(dir=record_signals_dir)

            (
                (abp_signals, non_abp_signals),
                n_non_abp_initial_samples_cut
            ) = align_signal_banks_and_remove_invalid_trailing_segments(
                abp_signal_bank=abp_signals,
                non_abp_signal_bank=non_abp_signals,
                abp_shifts_in_s=records_info[record_id]['abp_shifts_in_s']
            )

            if INSPECTION:
                from matplotlib.pyplot import show, subplots
                from sklearn.preprocessing import scale
                a = abp_signals.copy()
                b = non_abp_signals.copy()

            (
                abp_signals,
                non_abp_signals
            ) = filter_signal_banks(
                abp_signal_bank=abp_signals,
                non_abp_signal_bank=non_abp_signals
            )

            if INSPECTION:
                print(subject_id, record_dir_name)
                while True:
                    try:
                        fig, axes = subplots(10, 2)
                        fig.tight_layout()
                        i = input("from:to <- ")
                        if i == '':
                            break
                        i, j = i.split(':')
                        fig.suptitle(subject_id + ', ' + record_dir_name + ', ' + i + ':' + j)
                        i, j = map(lambda x: int(float(x) * 60 * SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ), [i, j])
                        print(i, ':', j, 'over', b[INDEX_FOR_ECG].shape[0])
                        axis = axes[0, 0]
                        axis.plot(scale(b[INDEX_FOR_ECG][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_ECG][i:j].astype(dtype='float64')))
                        axis.set_title('ECG')
                        axis = axes[0, 1]
                        axis.plot(scale(b[INDEX_FOR_PPG_GREEN][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_PPG_GREEN][i:j].astype(dtype='float64')))
                        axis.set_title('PPG G')
                        axis = axes[1, 0]
                        axis.plot(scale(b[INDEX_FOR_PPG_INFRARED][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_PPG_INFRARED][i:j].astype(dtype='float64')))
                        axis.set_title('PPG I')
                        axis = axes[1, 1]
                        axis.plot(scale(b[INDEX_FOR_PPG_RED][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_PPG_RED][i:j].astype(dtype='float64')))
                        axis.set_title('PPG R')
                        axis = axes[2, 0]
                        axis.plot(scale(b[INDEX_FOR_ACOUSTIC][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_ACOUSTIC][i:j].astype(dtype='float64')))
                        axis.set_title('ACOUSTIC')
                        axis = axes[2, 1]
                        axis.plot(scale(b[INDEX_FOR_BIOIMPEDANCE][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_BIOIMPEDANCE][i:j].astype(dtype='float64')))
                        axis.set_title('BIOIMPEDANCE')
                        axis = axes[3, 0]
                        axis.plot(scale(b[INDEX_FOR_ACC_XYZ_MAGN][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_ACC_XYZ_MAGN][i:j].astype(dtype='float64')))
                        axis.set_title('ACC M')
                        axis = axes[3, 1]
                        axis.plot(scale(b[INDEX_FOR_ACC_X][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_ACC_X][i:j].astype(dtype='float64')))
                        axis.set_title('ACC X')
                        axis = axes[4, 0]
                        axis.plot(scale(b[INDEX_FOR_ACC_Y][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_ACC_Y][i:j].astype(dtype='float64')))
                        axis.set_title('ACC Y')
                        axis = axes[4, 1]
                        axis.plot(scale(b[INDEX_FOR_ACC_Z][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_ACC_Z][i:j].astype(dtype='float64')))
                        axis.set_title('ACC Z')
                        axis = axes[5, 0]
                        axis.plot(scale(b[INDEX_FOR_SCG][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_SCG][i:j].astype(dtype='float64')))
                        axis.set_title('SCG')
                        axis = axes[5, 1]
                        axis.plot(scale(b[INDEX_FOR_TEMP_BODY][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_TEMP_BODY][i:j].astype(dtype='float64')))
                        axis.set_title('TEMP BODY')
                        axis = axes[6, 0]
                        axis.plot(scale(b[INDEX_FOR_TEMP_EXT][i:j].astype(dtype='float64')))
                        axis.plot(scale(non_abp_signals[INDEX_FOR_TEMP_EXT][i:j].astype(dtype='float64')))
                        axis.set_title('TEMP EXT')
                        axis = axes[6, 1]
                        axis.plot(scale(non_abp_signals[INDEX_FOR_RESP_PT_1][i:j].astype(dtype='float64')))
                        axis.set_title('RESP 1')
                        axis = axes[7, 0]
                        axis.plot(scale(non_abp_signals[INDEX_FOR_RESP_PT_2][i:j].astype(dtype='float64')))
                        axis.set_title('RESP 2')
                        axis = axes[7, 1]
                        axis.plot(scale(non_abp_signals[INDEX_FOR_RESP_PT_3][i:j].astype(dtype='float64')))
                        axis.set_title('RESP 3')
                        axis = axes[8, 0]
                        axis.set_visible(False)  # blank
                        i, j = map(lambda x: int(x / SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ * SAMPLING_FREQUENCY_OF_IN_HOUSE_NON_ABP_SIGNALS_IN_HZ), [i, j])
                        axis = axes[8, 1]
                        axis.plot(scale(a[INDEX_FOR_ABP_MASK][i:j].astype(dtype='float64')))
                        axis.plot(scale(abp_signals[INDEX_FOR_ABP_MASK][i:j].astype(dtype='float64')))
                        axis.set_title('ABP MASK')
                        axis = axes[9, 0]
                        axis.plot(scale(a[INDEX_FOR_BRACHIAL_ABP][i:j].astype(dtype='float64')))
                        axis.plot(scale(abp_signals[INDEX_FOR_BRACHIAL_ABP][i:j].astype(dtype='float64')))
                        axis.set_title('BR ABP')
                        axis = axes[9, 1]
                        axis.plot(scale(a[INDEX_FOR_FINGER_ABP][i:j].astype(dtype='float64')))
                        axis.plot(scale(abp_signals[INDEX_FOR_FINGER_ABP][i:j].astype(dtype='float64')))
                        axis.set_title('FI ABP')
                        show()
                    except Exception as exception:
                        print(exception)

            else:
                downsample_abp_related_and_segment_and_save_segments_to_sets(
                    abp_signal_bank=abp_signals,
                    non_abp_signal_bank=non_abp_signals,
                    resting_periods=records_info[record_id]['resting_periods'],
                    resting_periods_alignment_offset=n_non_abp_initial_samples_cut,
                    subject=subject_id,
                    record_date=record_dir_name,
                    annotation_mode=is_annotation_mode,
                    partitioning_modality=dataset_partitioning_modality,
                    partitioning_configurations=dataset_partitioning_configurations
                )
