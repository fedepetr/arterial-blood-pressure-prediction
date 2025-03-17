"""
Script to visualize the saved signal kernels.
"""


from os import getcwd
from os.path import join as path_join
from sys import path

from matplotlib.pyplot import (
    show,
    subplots,
)
from numpy import load

path.append(path_join(getcwd(), 'modeling'))
from constants import (
    PATH_TO_KERNEL_FOR_ABP,
    PATH_TO_KERNEL_FOR_ECG_LEAD,
    PATH_TO_KERNEL_FOR_PPG
)


def load_and_plot_kernels():
    """
    Load and plot all of the signal kernels.
    """
    figure, axes = subplots(3)
    figure.suptitle(t="Kernels")

    for subplot_index, kernel_path_and_name in enumerate(
        [
                (PATH_TO_KERNEL_FOR_ABP, 'ABP'),
                (PATH_TO_KERNEL_FOR_ECG_LEAD, 'ECG (Lead II)'),
                (PATH_TO_KERNEL_FOR_PPG, 'PPG')
        ]
    ):
        kernel_path, kernel_name = kernel_path_and_name

        kernel = load(kernel_path)
        x_axis_values = list(range(kernel.shape[0]))

        axes[subplot_index].plot(x_axis_values, kernel)
        axes[subplot_index].title.set_text(s=kernel_name)

    show()


if __name__ == '__main__':
    load_and_plot_kernels()
