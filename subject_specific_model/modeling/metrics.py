"""
Definition of custom metrics of interest.
"""


# pylint: disable=no-name-in-module,import-error
from tensorflow import cast, convert_to_tensor, Tensor
from tensorflow.math import reduce_mean
# pylint: enable=no-name-in-module,import-error


def mean_signed_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Definition of mean signed (i.e. non-absolute) error.
    """
    y_pred = convert_to_tensor(value=y_pred)
    y_true = cast(x=y_true, dtype=y_pred.dtype)
    return reduce_mean(input_tensor=(y_pred - y_true), axis=-1)
