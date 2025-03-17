"""
Analysis of dataset labels.
"""


from typing import Any, Dict, List, Tuple

from matplotlib.pyplot import show, subplots_adjust
from mpl_toolkits.axes_grid1 import host_subplot
from scipy.stats import norm
# pylint: disable=no-name-in-module,import-error
from tensorflow.data import Dataset
# pylint: enable=no-name-in-module,import-error
from yaml import dump as yaml_dump

from constants import (
    ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP,
    LABELS_DISTRIBUTIONS_FITTING_PARAMETERS_FILE,
    MINI_BATCH_SIZE,
    OUTPUT_NAME_FOR_DBP,
    OUTPUT_NAME_FOR_SBP
)
from samples_and_labels import create_datasets, TASK


def collect_labels_values_over_dataset(dataset: Dataset) -> None:
    """
    Collect all values of each label across the given dataset by adding them
    to the respective label global list.
    """
    global dbp_values
    global sbp_values
    for mini_batch_data in dataset:
        mini_batch_labels = mini_batch_data[1]
        dbp_values.extend(
            mini_batch_labels[OUTPUT_NAME_FOR_DBP].numpy().flatten().tolist()
        )
        sbp_values.extend(
            mini_batch_labels[OUTPUT_NAME_FOR_SBP].numpy().flatten().tolist()
        )


def plot_label_distribution(
        label_values: List[float],
        name: str,
        fitting_distribution_parameters: Dict[str, float]
) -> Tuple[List[int], List[float], Any]:
    """
    Plot the distribution of the given label values with the fitting curve.
    """
    histogram_color = 'dodgerblue'
    distribution_color = 'mediumslateblue'
    weights_color = 'limegreen'

    host_plot = host_subplot(111)
    subplots_adjust(right=0.75)
    parasite_plot_for_distribution = host_plot.twinx()
    parasite_plot_for_weights = host_plot.twinx()

    bin_counts, bin_boundaries, _ = host_plot.hist(
        x=label_values,
        bins=100,
        color=histogram_color
    )
    parasite_plot_for_distribution.plot(
        bin_boundaries,
        norm.pdf(x=bin_boundaries, **fitting_distribution_parameters),
        color=distribution_color
    )
    parasite_plot_for_weights.plot(
        bin_boundaries,
        (
            1 - (
                norm.pdf(x=bin_boundaries, **fitting_distribution_parameters)
                / norm.pdf(
                    x=fitting_distribution_parameters['loc'],
                    **fitting_distribution_parameters
                )
            )
        ),
        color=weights_color
    )

    host_plot.set_title(label=f'{name} Distribution')
    host_plot.set_xlabel(xlabel='mmHg')
    host_plot.set_ylabel(ylabel='counts')
    host_plot.axis['left'].label.set_color(histogram_color)
    parasite_plot_for_distribution.set_ylabel(ylabel='probability density')
    parasite_plot_for_distribution.axis['right'].label.set_color(distribution_color)
    parasite_plot_for_weights.spines.right.set_position(('axes', 1.2))
    # parasite_plot_for_weights.set_ylim(bottom=0, top=1000)
    parasite_plot_for_weights.set_ylabel(ylabel='label weights')
    parasite_plot_for_weights.axis['right'].label.set_color(weights_color)

    show()

    return bin_counts, bin_boundaries


if __name__ == '__main__':
    if TASK != ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP:
        raise NotImplementedError(
            f"Analysis only for {ARGUMENT_NAME_FOR_TASK_AS_DBP_AND_SBP} task."
        )

    training_set, validation_set, test_set = create_datasets(
        mini_batch_size=MINI_BATCH_SIZE
    )

    dbp_values = []
    sbp_values = []
    collect_labels_values_over_dataset(dataset=training_set)
    del training_set
    # collect_labels_values_over_dataset(dataset=validation_set)
    del validation_set
    # collect_labels_values_over_dataset(dataset=test_set)
    del test_set

    dbp_training_gaussian_distribution = {}
    (
        dbp_training_gaussian_distribution['loc'],
        dbp_training_gaussian_distribution['scale']
    ) = norm.fit(data=dbp_values)
    sbp_training_gaussian_distribution = {}
    (
        sbp_training_gaussian_distribution['loc'],
        sbp_training_gaussian_distribution['scale']
    ) = norm.fit(data=sbp_values)

    print(OUTPUT_NAME_FOR_DBP.upper() + " Gaussian Distribution Parameters:", end='')
    print('\t', dbp_training_gaussian_distribution)
    print(OUTPUT_NAME_FOR_SBP.upper() + " Gaussian Distribution Parameters:", end='')
    print('\t', sbp_training_gaussian_distribution)

    _ = plot_label_distribution(
        label_values=dbp_values,
        name=OUTPUT_NAME_FOR_DBP.upper() + " Over Training Set",
        fitting_distribution_parameters=dbp_training_gaussian_distribution
    )
    _ = plot_label_distribution(
        label_values=sbp_values,
        name=OUTPUT_NAME_FOR_SBP.upper() + " Over Training Set",
        fitting_distribution_parameters=sbp_training_gaussian_distribution
    )

    with open(LABELS_DISTRIBUTIONS_FITTING_PARAMETERS_FILE, 'w') as file:
        yaml_dump(
            data=[
                dbp_training_gaussian_distribution['loc'].item(),
                dbp_training_gaussian_distribution['scale'].item(),
                sbp_training_gaussian_distribution['loc'].item(),
                sbp_training_gaussian_distribution['scale'].item()
            ],
            stream=file
        )
