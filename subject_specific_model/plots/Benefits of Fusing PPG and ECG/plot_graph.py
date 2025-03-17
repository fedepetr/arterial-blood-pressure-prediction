"""
"""


from json import load as json_load

from matplotlib import rcParams
from matplotlib.pyplot import figure, grid, legend, plot, show, title, ylim


rcParams.update({'font.size': 16})

# for filename in FILENAMES:  # = input("File? ")

def plot_file_data(filename: str, plot_name: str, plot_color: str) -> None:
    with open(filename, 'r') as file:
        content = json_load(file)

    epoch_indexes = []
    values = []
    for data_point in content:
        epoch_indexes.append(data_point[1])
        values.append(data_point[2])

    plot(epoch_indexes, values, color=plot_color, label=plot_name)


figure()

plot_file_data(
    filename='N_ppg_Y_ecg___overfitting_100_epochs_train.json',
    plot_name="ECG only - Training Set",
    plot_color='limegreen'
)

plot_file_data(
    filename='N_ppg_Y_ecg___overfitting_100_epochs_validation.json',
    plot_name="ECG only - Validation Set",
    plot_color='lawngreen'
)

plot_file_data(
    filename='Y_ppg_N_ecg___overfitting_100_epochs_train.json',
    plot_name="PPG only - Training Set",
    plot_color='orchid'
)

plot_file_data(
    filename='Y_ppg_N_ecg___overfitting_100_epochs_validation.json',
    plot_name="PPG only - Validation Set",
    plot_color='magenta'
)

plot_file_data(
    filename='Y_ppg_Y_ecg___overfitting_100_epochs_train.json',
    plot_name="PPG + ECG - Training Set",
    plot_color='deepskyblue'
)

plot_file_data(
    filename='Y_ppg_Y_ecg___overfitting_100_epochs_validation.json',
    plot_name="PPG + ECG - Validation Set",
    plot_color='cyan'
)

ylim((100, 500))
grid(axis='y')
legend(loc='lower left')
title("MSE Loss with Epochs", fontdict={'size': 39})

show()
