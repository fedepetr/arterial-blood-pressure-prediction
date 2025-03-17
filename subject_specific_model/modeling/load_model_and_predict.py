"""
Script to load a pretrained model and employ it to run interactive predictions
via CLI.
"""
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
from scipy import signal as sg
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QVBoxLayout,
    QLineEdit,
    QWidget,
    QPushButton,
)


import json
from os import listdir
from os.path import join as path_join
from random import choice, seed
import tensorflow as tf
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

from PyQt6.QtWidgets import QMainWindow
global indice
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Prediction Visualization')
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
         
        global indice 
        indice = 0 

        #Button fuction definition 
        def update_data(self):
            global indice
            indice = indice + 1
            print(indice)
            predicted_signal = model.predict_on_batch(
            x=samples[indice]
            ).squeeze()
            grafico_p.setData(x_axis_values,predicted_signal)
            grafico_t.setData(x_axis_values,labels[indice][0])
            
            segnale_ppg = samples[indice][ :, :, 11]
            segnale_ecg = samples[indice][ :, :, 9]
            
            grafico_ppg.setData(x_axis_values, segnale_ppg[0])
            grafico_ecg.setData(x_axis_values, segnale_ecg[0])

        def back_data(self):
            global indice
            indice = indice - 1
            
            predicted_signal = model.predict_on_batch(
            x=samples[indice]
            ).squeeze()

            segnale_ppg = samples[indice][ :, :, 11]
            segnale_ecg = samples[indice][ :, :, 9]
            grafico_p.setData(x_axis_values,predicted_signal)
            grafico_t.setData(x_axis_values,labels[indice][0])
            grafico_ppg.setData(x_axis_values, segnale_ppg[0])
            grafico_ecg.setData(x_axis_values, segnale_ecg[0])

        #Load the data
        samples = []
        labels = []
        mask_r = []
        count = 0  
        
        
        for mini_batch_data in chosen_set:
                print('samples', tf.shape(mini_batch_data[0]))    
                print('labels', tf.shape(mini_batch_data[1]))    
                print('mask', tf.shape(mini_batch_data[2]))    
                mini_batch_samples = mini_batch_data[0]
                mini_batch_labels = mini_batch_data[1]
                mini_batch_mask_as_weights = mini_batch_data[2]
                samples.append(mini_batch_samples)
                labels.append(mini_batch_labels)
                mask_r.append(mini_batch_mask_as_weights)
                count = count + 1

        
        # Definition of the predicted signal 
        predicted_signal = model.predict_on_batch(
            x=samples[indice]
        ).squeeze()
                
        x_axis_values = (
                array(list(range(len(labels[0][0])))) /
                SIGNALS_SAMPLING_FREQUENCY
            )
        # print(len(labels), count)
        # plt.plot(x_axis_values ,predicted_signal)
        # plt.plot(x_axis_values , labels[34][0])
        # plt.show()
            

        #Widget definition

        #target plot
        self.target_graph = pg.PlotWidget()
        self.target_graph.setTitle("Predicted signal Vs Target signal", color="b", size="30pt")
        self.target_graph.setBackground('w')
        pen_r = pg.mkPen(color=(230, 0, 0), width=3)
        pen_b = pg.mkPen(color=(0, 0, 230), width=3)
        styles = {'color':'r', 'font-size':'20px'}
        self.target_graph.setLabel('left', 'mmHg', **styles)
        self.target_graph.setLabel('bottom', 'Seconds (s)', **styles)
        self.target_graph.showGrid(x=True, y=True)
        grafico_p = self.target_graph.plot(x_axis_values, predicted_signal, 'predicted signal', pen=pen_r)
        grafico_t = self.target_graph.plot(x_axis_values, labels[indice][0][0], 'target',pen = pen_b)
        

        #ppg plot
        self.ppg_graph = pg.PlotWidget()
        self.ppg_graph.setTitle("PPG signal", color="b", size="30pt")
        self.ppg_graph.setBackground('w')
        pen_r = pg.mkPen(color=(230, 0, 0), width=3)
        pen_b = pg.mkPen(color=(0, 0, 230), width=3)
        styles = {'color':'r', 'font-size':'20px'}
        self.ppg_graph.setLabel('left', 'A.U.', **styles)
        self.ppg_graph.setLabel('bottom', 'Seconds (s)', **styles)
        self.ppg_graph.showGrid(x=True, y=True)
        grafico_ppg = self.ppg_graph.plot(x_axis_values, samples[0][:,:,13], 'predicted signal', pen=pen_r)
        self.ppg_graph.addLegend()

        #ecg plot
        self.ecg_graph = pg.PlotWidget()
        self.ecg_graph.setTitle("ECG signal ", color="b", size="30pt")
        self.ecg_graph.setBackground('w')
        pen_r = pg.mkPen(color=(230, 0, 0), width=3)
        pen_b = pg.mkPen(color=(0, 0, 230), width=3)
        styles = {'color':'r', 'font-size':'20px'}
        self.ecg_graph.setLabel('left', 'mmHg', **styles)
        self.ecg_graph.setLabel('bottom', 'Seconds (s)', **styles)
        self.ecg_graph.showGrid(x=True, y=True)
        grafico_ecg = self.ecg_graph.plot(x_axis_values, labels[indice][0], 'target',pen = pen_b)
        self.ecg_graph.addLegend()
        
        
        #Button Widget
        self.forward_button = QPushButton('forward')
        self.back_button = QPushButton('back')
        self.forward_button.clicked.connect(update_data)
        self.back_button.clicked.connect(back_data)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.target_graph)
        self.layout.addWidget(self.ppg_graph)
        self.layout.addWidget(self.ecg_graph)
        self.layout.addWidget(self.forward_button)
        self.layout.addWidget(self.back_button)
        centralWidget.setLayout(self.layout)

        

        


def main():
    """PyCalc's main function."""
    pycalcApp = QApplication([])
    pycalcWindow = Window()
    pycalcWindow.show()
    sys.exit(pycalcApp.exec())



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

        if True:
            main()
            
        count = 0 
        dataset = []

        if False :
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

                    predicted_signal = tf.make_tensor_proto(predicted_signal)  # convert `tensor a` to a proto tensor
                    predicted_signal = tf.make_ndarray(predicted_signal).tolist()
                    target_signal = mini_batch_labels[0].numpy().tolist()
                    mask = mini_batch_mask_as_weights[0].numpy()
                    mask = mask.flatten().astype(int).tolist()

                    vocabolario = {"id": count, "predicted": predicted_signal, "target": target_signal , "mask": mask }
                    dataset.append(vocabolario)
                
                count = count + 1
                print(count)
        
            dataset = json.dumps(dataset)

            # Salvataggio dei segnali JSON su un file
            with open(MODEL_FILE_PATH  + ".json", "w") as file:
                file.write(dataset)

            print("Segnali salvati con successo.")
        
        if False:

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
