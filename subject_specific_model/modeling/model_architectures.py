"""
Model architecture definition.
"""


from math import log
from os.path import join as path_join
from typing import Any, get_args, Literal, Optional, Type

# pylint: disable=no-name-in-module,import-error
from tensorflow import expand_dims, squeeze
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    AveragePooling1D,
    BatchNormalization,
    # Bidirectional,
    Concatenate,
    Convolution1D,
    Convolution2D,
    Convolution1DTranspose,
    Convolution2DTranspose,
    Dense,
    Flatten,
    GRU,
    Lambda,
    MaxPooling1D,
    Normalization,
    Permute
)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import plot_model
# pylint: enable=no-name-in-module,import-error

from constants import (
    ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE,
    ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME,
    ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY,
    ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM,
    DOUBLE_UNET_VARIANT,
    DOUBLE_UNET_VARIANT_SUBMODEL_NAMES,
    ACTIVATION_FUNCTION,
    BATCH_NORMALIZATION,
    BATCH_NORMALIZATION_AXIS,
    DATA_TYPE_FOR_INPUTS,
    DOWN_AND_UP_SAMPLING_SPAN,
    FREQUENCY_BIN_AVERAGING_DOWNSAMPLING_SPAN,
    INITIALLY_SEPARATE_BRANCHES_IN_ENCODER_ONLY_MODEL,
    INPUTS_STANDARDIZATION_LAYER_NAME,
    MODELING_DIR_NAME,
    MULTI_RES_UNET_GRU_AFTER_BOTTLENECK,
    MULTI_RES_UNET_GRU_AFTER_ENCODER,
    MULTI_RES_UNET_GRU_AFTER_DECODER,
    MULTI_RES_UNET_GRU_ALONG_SKIP_CONNECTIONS,
    MULTI_RES_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
    MULTI_RES_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS,
    MULTI_RES_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER,
    MULTI_RES_UNET_SKIP_CONNECTIONS,
    N_DENSE_WEIGHTS_DROP_FACTOR,
    N_FILTERS_IN_FIRST_DECODER_LAYER,
    N_FILTERS_MULTIPLICATION_FACTOR,
    N_FREQUENCY_BIN_AVERAGING_STEPS,
    N_FULLY_CONNECTED_LAYERS,
    N_IN_HOUSE_INPUT_CHANNELS,
    N_SAMPLES_PER_IN_HOUSE_SEGMENT,
    N_SPECTROGRAM_FREQUENCIES,
    N_SPECTROGRAM_SAMPLES,
    N_WEIGHTS_IN_FIRST_ENCODER_LAYER,
    N_UCI_SIGNAL_SEGMENT_SAMPLES,
    L2_REGULARIZATION_LAMBDA,
    OUTPUT_NAME_FOR_DBP,
    OUTPUT_NAME_FOR_SBP,
    RESULTS_DIR_NAME,
    ROOT_DIR,
    VANILLA_UNET_GRU_AFTER_BOTTLENECK,
    VANILLA_UNET_GRU_AFTER_DECODER,
    VANILLA_UNET_GRU_AFTER_ENCODER,
    VANILLA_UNET_GRU_ALONG_SKIP_CONNECTIONS,
    VANILLA_UNET_MULTI_BRANCH_ENCODER,
    VANILLA_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
    VANILLA_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS,
    VANILLA_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER,
    VANILLA_UNET_SKIP_CONNECTIONS,
    ValidArgumentsForDataSource,
    ValidArgumentsForInputDomain,
    ValidArgumentsForTask,
    get_required_arguments
)


PADDING_MODE = 'same'
COMMON_CONVOLUTION_KWARGS = {
    'kernel_size': 6,
    'activation': 'linear',
    'padding': PADDING_MODE
}
COMMON_FULLY_CONNECTED_KWARGS = {
    'activation': 'linear',
}
REGULARIZATION_KWARGS = {
    'kernel_regularizer': L2(l2=L2_REGULARIZATION_LAMBDA),
    'bias_regularizer': L2(l2=L2_REGULARIZATION_LAMBDA)
}


ValidArgumentsForConvolutionDimension = Literal[1, 2]


def build_enriched_convolutional_layer(
        convolution_dimension: int,
        n_filters: int,
        transposed: bool = False,
        kernel_size_on_frequency_axis: Optional[int] = None,
        upsampling_span: Optional[int] = None
) -> Sequential:
    """
    Return an instance of a block of layers based on either a convolutional
    unit or a transposed-convolution unit with batch normalization between the
    (transposed) convolution result and the activation function.
    """
    assert (
        convolution_dimension in get_args(
            tp=ValidArgumentsForConvolutionDimension
        )
    ), "Invalid 'convolution_dimension' argument passed."

    convolutional_layer_kwargs = {'filters': n_filters}
    convolutional_layer_kwargs.update(COMMON_CONVOLUTION_KWARGS)
    convolutional_layer_kwargs.update(REGULARIZATION_KWARGS)
    if convolution_dimension == 2:
        convolutional_layer_kwargs['padding'] = 'valid'
        convolutional_layer_kwargs['kernel_size'] = (
            kernel_size_on_frequency_axis,
            COMMON_CONVOLUTION_KWARGS['kernel_size']
        )
    if transposed:
        convolutional_layer_kwargs['strides'] = upsampling_span

    stacked_layers = [
            (
                Convolution1D(**convolutional_layer_kwargs) if not transposed
                else Convolution1DTranspose(**convolutional_layer_kwargs)
            ) if convolution_dimension == 1 else (
                Convolution2D(**convolutional_layer_kwargs) if not transposed
                else Convolution2DTranspose(**convolutional_layer_kwargs)
            ),
            Activation(activation=ACTIVATION_FUNCTION)
        ]
    if BATCH_NORMALIZATION:
        stacked_layers.insert(
            1,
            BatchNormalization(axis=BATCH_NORMALIZATION_AXIS)
        )
    return Sequential(layers=stacked_layers)


def build_enriched_fully_connected_layer(
        n_outputs: int,
        layer_name: Optional[str] = None
) -> Sequential:
    """
    Return an instance of a block of layers based on fully-connected unit with
    batch normalization between the dense sum of products' result and the
    activation function.
    """
    stacked_layers = [
        Dense(
            units=n_outputs,
            **COMMON_FULLY_CONNECTED_KWARGS,
            **REGULARIZATION_KWARGS
        ),
        Activation(activation=ACTIVATION_FUNCTION),
    ]
    if BATCH_NORMALIZATION:
        stacked_layers.insert(
            1,
            BatchNormalization(axis=BATCH_NORMALIZATION_AXIS)
        )
    return Sequential(
        layers=stacked_layers,
        name=layer_name
    )


def build_1d_fcn_encoder_decoder_architecture(  # pylint: disable=unused-argument
        n_convolutional_layers_per_block: int,
        n_encoder_blocks: int,
        n_filters_in_first_layer: int,
        in_house_data: bool,
        multi_branch_encoder: bool,
        multi_res_variant: bool,
        recurrent_encoder: bool,
        recurrent_bottleneck: bool,
        recurrent_decoder: bool,
        recurrent_skip_connections: bool,
        skip_connections: bool,
        **unrelevant_kwargs: Any
) -> Type[Model]:
    """
    Return an instance of a fully-convolutional network (FCN) encoder-decoder
    model architecture for signal prediction from 1-dimensional input signals
    representing time series.
    """
    if in_house_data:
        signals = Input(
            shape=(
                N_SAMPLES_PER_IN_HOUSE_SEGMENT,
                1 if multi_res_variant else N_IN_HOUSE_INPUT_CHANNELS
            ),
            dtype=DATA_TYPE_FOR_INPUTS
        )
    else:
        if multi_res_variant:
            raise NotImplementedError()
        signals = Input(
            shape=(N_UCI_SIGNAL_SEGMENT_SAMPLES, 2),
            dtype=DATA_TYPE_FOR_INPUTS
        )

    current_outputs = signals

    current_outputs = Normalization(
        axis=-1,
        name=INPUTS_STANDARDIZATION_LAYER_NAME
    )(current_outputs)

    # NOTE: code snippet left in case useful:
    #    -    -    -    -    -    -    -    -    -    -    -    -    -    -
    # # considering only the ECG channel:
    # current_outputs = Lambda(
    #     function=lambda outputs: expand_dims(
    #         input=outputs[:, :, 6],
    #         axis=-1
    #     )
    # )(current_outputs)
    # # considering only the PPG channels:
    # current_outputs = Lambda(
    #     function=lambda outputs: outputs[:, :, 7:10]
    # )(current_outputs)
    #    -    -    -    -    -    -    -    -    -    -    -    -    -    -

    # ------------------------------------------------------------------------
    # ENCODER
    # ------------------------------------------------------------------------

    if skip_connections:
        outputs_transmitted_through_skip_connections = []
    current_n_convolutional_filters = (
        n_filters_in_first_layer / N_FILTERS_MULTIPLICATION_FACTOR
    )

    # encoder with multiple branches:
    if multi_branch_encoder:
        if multi_res_variant:
            raise NotImplementedError(
                "Multi-branch encoder not allowed (as it would be " +
                "meaningless) with MultiResUNet, which is here employed " +
                "for single-channel inputs."
            )

        branches_current_outputs = [
            # high-frequency channels:
            Lambda(function=lambda outputs: outputs[:, :, 0:1])(current_outputs),
            # medium-frequency channels:
            Lambda(function=lambda outputs: outputs[:, :, 1:11])(current_outputs),
            # low-frequency channels:
            Lambda(function=lambda outputs: outputs[:, :, 11:16])(current_outputs),
        ]
        # for each convolutional block of the encoder:
        for _ in range(n_encoder_blocks):
            current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

            if skip_connections:
                branch_outputs_for_skip_connections = [None] * 3

            # for each branch:
            for branch_index, current_outputs in enumerate(branches_current_outputs):
                # processing via convolutions:
                for _ in range(n_convolutional_layers_per_block):
                    enriched_convolutional_layer = build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                    current_outputs = enriched_convolutional_layer(current_outputs)

                if skip_connections:
                    # remembering branch activations for merging them with all
                    # other branches' ones, for skip connections:
                    branch_outputs_for_skip_connections[branch_index] = current_outputs

                # downsampling:
                pooling_layer = MaxPooling1D(
                    pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                    padding=PADDING_MODE
                )
                current_outputs = pooling_layer(current_outputs)

                branches_current_outputs[branch_index] = current_outputs

            if skip_connections:
                # merging activations of all branches and remembering them for
                # skip connections:
                outputs_transmitted_through_skip_connections.append(
                    Concatenate(axis=-1)(branch_outputs_for_skip_connections)
                )

        current_outputs = Concatenate(axis=-1)(branches_current_outputs)

    # encoder with a single branch:
    else:
        # for each convolutional block of the encoder:
        for encoder_block_index in range(n_encoder_blocks):
            current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

            # multi-res variant:
            if multi_res_variant:
                alpha = 1.67
                layer_filters_fractions = [1/6, 1/3, 1/2]
                outputs_of_residual_connection = []

                initial_current_outputs = current_outputs

                for filters_fraction in layer_filters_fractions:
                    enriched_convolutional_layer = build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=round(
                            current_n_convolutional_filters * alpha * filters_fraction
                        )
                    )
                    current_outputs = enriched_convolutional_layer(current_outputs)
                    outputs_of_residual_connection.append(current_outputs)

                current_outputs = Concatenate(axis=-1)(
                    outputs_of_residual_connection
                )
                if BATCH_NORMALIZATION:
                    current_outputs = BatchNormalization(
                        axis=BATCH_NORMALIZATION_AXIS
                    )(current_outputs)

                residual_outputs = Convolution1D(
                    filters=current_outputs.shape[-1],
                    **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
                    **REGULARIZATION_KWARGS
                )(initial_current_outputs)
                if BATCH_NORMALIZATION:
                    residual_outputs = BatchNormalization(
                        axis=BATCH_NORMALIZATION_AXIS
                    )(residual_outputs)
                # TODO:
                # residual_outputs = Activation(activation= 'relu')(residual_outputs)

                if BATCH_NORMALIZATION:
                    current_outputs = BatchNormalization(
                        axis=BATCH_NORMALIZATION_AXIS
                    )(current_outputs)
                current_outputs = current_outputs + residual_outputs
                # TODO:
                # current_outputs = Activation(activation= 'relu')(current_outputs)
                if BATCH_NORMALIZATION:
                    current_outputs = BatchNormalization(
                        axis=BATCH_NORMALIZATION_AXIS
                    )(current_outputs)

                # res path for skip connections:
                res_outputs = current_outputs
                for _ in range(n_encoder_blocks - encoder_block_index):
                    external_outputs = Convolution1D(
                        filters=current_n_convolutional_filters,
                        **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
                        **REGULARIZATION_KWARGS
                    )(res_outputs)
                    if BATCH_NORMALIZATION:
                        external_outputs = BatchNormalization(
                            axis=BATCH_NORMALIZATION_AXIS
                        )(external_outputs)
                    # TODO:
                    # external_outputs = Activation(activation= 'relu')(external_outputs)

                    enriched_convolutional_layer = build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                    res_outputs = enriched_convolutional_layer(res_outputs)

                    if BATCH_NORMALIZATION:
                        res_outputs = BatchNormalization(
                            axis=BATCH_NORMALIZATION_AXIS
                        )(res_outputs)
                    res_outputs = res_outputs + external_outputs
                    # TODO:
                    # res_output = Activation(activation= 'relu')(res_output)
                    if BATCH_NORMALIZATION:
                        res_outputs = BatchNormalization(
                            axis=BATCH_NORMALIZATION_AXIS
                        )(res_outputs)

                if skip_connections:
                    # remembering activations for skip connections:
                    outputs_transmitted_through_skip_connections.append(res_outputs)

                # downsampling:
                pooling_layer = MaxPooling1D(
                    pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                    padding=PADDING_MODE
                )
                current_outputs = pooling_layer(current_outputs)

            # non-multi-res variant:
            else:
                # processing via convolutions:
                for _ in range(n_convolutional_layers_per_block):
                    enriched_convolutional_layer = build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                    current_outputs = enriched_convolutional_layer(current_outputs)

                if skip_connections:
                    # remembering activations for skip connections:
                    outputs_transmitted_through_skip_connections.append(current_outputs)

                # downsampling:
                pooling_layer = MaxPooling1D(
                    pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                    padding=PADDING_MODE
                )
                current_outputs = pooling_layer(current_outputs)

    if recurrent_encoder:
        current_outputs = GRU(
            units=int(current_n_convolutional_filters),
            activation='linear',
            recurrent_activation='sigmoid',
            return_sequences=True,
            **REGULARIZATION_KWARGS
        )(current_outputs)

    # ------------------------------------------------------------------------
    # BOTTLENECK
    # ------------------------------------------------------------------------

    current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

    # multi-res variant:
    if multi_res_variant:
        alpha = 1.67
        layer_filters_fractions = [1/6, 1/3, 1/2]
        outputs_of_residual_connection = []

        initial_current_outputs = current_outputs

        for filters_fraction in layer_filters_fractions:
            enriched_convolutional_layer = build_enriched_convolutional_layer(
                convolution_dimension=1,
                n_filters=round(
                    current_n_convolutional_filters * alpha * filters_fraction
                )
            )
            current_outputs = enriched_convolutional_layer(current_outputs)
            outputs_of_residual_connection.append(current_outputs)

        current_outputs = Concatenate(axis=-1)(
            outputs_of_residual_connection
        )
        if BATCH_NORMALIZATION:
            current_outputs = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(current_outputs)

        residual_outputs = Convolution1D(
            filters=current_outputs.shape[-1],
            **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
            **REGULARIZATION_KWARGS
        )(initial_current_outputs)
        if BATCH_NORMALIZATION:
            residual_outputs = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(residual_outputs)
        # TODO:
        # residual_outputs = Activation(activation= 'relu')(residual_outputs)

        if BATCH_NORMALIZATION:
            current_outputs = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(current_outputs)
        current_outputs = current_outputs + residual_outputs
        # TODO:
        # current_outputs = Activation(activation= 'relu')(current_outputs)
        if BATCH_NORMALIZATION:
            current_outputs = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(current_outputs)

    # non-multi-res variant:
    else:
        # processing via convolutions:
        for _ in range(n_convolutional_layers_per_block):
            enriched_convolutional_layer = build_enriched_convolutional_layer(
                convolution_dimension=1,
                n_filters=current_n_convolutional_filters
            )
            current_outputs = enriched_convolutional_layer(current_outputs)

    if recurrent_bottleneck:
        current_outputs = GRU(
            units=int(current_n_convolutional_filters),
            activation='linear',
            recurrent_activation='sigmoid',
            return_sequences=True,
            **REGULARIZATION_KWARGS
        )(current_outputs)

        # TODO: fix this
        # return Model(
        #     inputs=[signals],
        #     outputs=current_outputs
        # )

    # ------------------------------------------------------------------------
    # DECODER
    # ------------------------------------------------------------------------

    # for each convolutional block of the decoder:
    for corresponding_encoder_block_index in reversed(
            range(n_encoder_blocks)  # number of decoder blocks as encoder
    ):
        current_n_convolutional_filters /= N_FILTERS_MULTIPLICATION_FACTOR

        # upsampling while processing via transposed convolutions:
        enriched_convolutional_layer = build_enriched_convolutional_layer(
            convolution_dimension=1,
            n_filters=current_n_convolutional_filters,
            transposed=True,
            upsampling_span=DOWN_AND_UP_SAMPLING_SPAN
        )
        current_outputs = enriched_convolutional_layer(current_outputs)

        if skip_connections:
            # concatenating outputs with previous activations through skip
            # connections:
            skip_connection_output = outputs_transmitted_through_skip_connections[
                corresponding_encoder_block_index
            ]
            if recurrent_skip_connections:
                skip_connection_output = GRU(
                    units=int(current_n_convolutional_filters),
                    activation='linear',
                    recurrent_activation='sigmoid',
                    return_sequences=True,
                    **REGULARIZATION_KWARGS
                )(skip_connection_output)
            current_outputs = Concatenate(axis=-1)(
                [
                    current_outputs,
                    skip_connection_output
                ]
            )

        # multi-res variant:
        if multi_res_variant:
            alpha = 1.67
            layer_filters_fractions = [1/6, 1/3, 1/2]
            outputs_of_residual_connection = []

            initial_current_outputs = current_outputs

            for filters_fraction in layer_filters_fractions:
                enriched_convolutional_layer = build_enriched_convolutional_layer(
                    convolution_dimension=1,
                    n_filters=round(
                        current_n_convolutional_filters * alpha * filters_fraction
                    )
                )
                current_outputs = enriched_convolutional_layer(current_outputs)
                outputs_of_residual_connection.append(current_outputs)

            current_outputs = Concatenate(axis=-1)(
                outputs_of_residual_connection
            )
            if BATCH_NORMALIZATION:
                current_outputs = BatchNormalization(
                    axis=BATCH_NORMALIZATION_AXIS
                )(current_outputs)

            residual_outputs = Convolution1D(
                filters=current_outputs.shape[-1],
                **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
                **REGULARIZATION_KWARGS
            )(initial_current_outputs)
            if BATCH_NORMALIZATION:
                residual_outputs = BatchNormalization(
                    axis=BATCH_NORMALIZATION_AXIS
                )(residual_outputs)
            # TODO:
            # residual_outputs = Activation(activation= 'relu')(residual_outputs)

            if BATCH_NORMALIZATION:
                current_outputs = BatchNormalization(
                    axis=BATCH_NORMALIZATION_AXIS
                )(current_outputs)
            current_outputs = current_outputs + residual_outputs
            # TODO:
            # current_outputs = Activation(activation= 'relu')(current_outputs)
            if BATCH_NORMALIZATION:
                current_outputs = BatchNormalization(
                    axis=BATCH_NORMALIZATION_AXIS
                )(current_outputs)

        # non-multi-res variant:
        else:
            # processing via convolutions:
            for _ in range(n_convolutional_layers_per_block):
                enriched_convolutional_layer = build_enriched_convolutional_layer(
                    convolution_dimension=1,
                    n_filters=current_n_convolutional_filters
                )
                current_outputs = enriched_convolutional_layer(current_outputs)

    # ------------------------------------------------------------------------

    current_outputs = Convolution1D(
        # a single output scalar is predicted at any timestep for punctual
        # regression:
        filters=1,
        **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
        **REGULARIZATION_KWARGS
    )(current_outputs)

    if recurrent_decoder:
        current_outputs = GRU(
            units=1,
            activation='linear',
            recurrent_activation='sigmoid',
            return_sequences=True,
            **REGULARIZATION_KWARGS
        )(current_outputs)

    return Model(
        inputs=[signals],
        outputs=current_outputs,
        name=(
            DOUBLE_UNET_VARIANT_SUBMODEL_NAMES[1]
            if multi_res_variant else
            DOUBLE_UNET_VARIANT_SUBMODEL_NAMES[0]
        )
    )


def build_1d_fcn_encoder_architecture(  # pylint: disable=unused-argument
        initially_separate_branches: bool,
        n_convolutional_layers_per_block: int,
        n_total_encoder_blocks: int,
        n_filters_in_first_layer: int,
        in_house_data: bool,
        **unrelevant_kwargs: Any
) -> Type[Model]:
    """
    Return an instance of a fully-convolutional network (FCN) encoder model
    architecture for scalars prediction from 1-dimensional input signals
    representing time series.
    """
    if in_house_data:
        signals = Input(
            shape=(N_SAMPLES_PER_IN_HOUSE_SEGMENT, 14),
            dtype=DATA_TYPE_FOR_INPUTS
        )
    else:
        signals = Input(
            shape=(N_UCI_SIGNAL_SEGMENT_SAMPLES, 2),
            dtype=DATA_TYPE_FOR_INPUTS
        )

    current_outputs = Normalization(
        axis=-1,
        name=INPUTS_STANDARDIZATION_LAYER_NAME
    )(signals)

    current_n_convolutional_filters = (
        n_filters_in_first_layer / N_FILTERS_MULTIPLICATION_FACTOR
    )

    if initially_separate_branches:
        # --------------------------------------------------------------------
        # ENCODER WITH INITIALLY-SEPARATE BRANCHES
        # --------------------------------------------------------------------

        n_encoder_blocks_with_separate_branches = round(
            n_total_encoder_blocks / 2
        )
        n_encoder_blocks_fusing_branches = (
            n_total_encoder_blocks - n_encoder_blocks_with_separate_branches
        )

        # --------------------------------------------------------------------
        # ENCODER - SEPARATE BRANCHES
        # --------------------------------------------------------------------

        current_ppg_representation = Lambda(
            function=lambda outputs: expand_dims(
                input=outputs[:, :, 0],
                axis=-1
            )
        )(current_outputs)

        current_ecg_representation = Lambda(
            function=lambda outputs: expand_dims(
                input=outputs[:, :, 1],
                axis=-1
            )
        )(current_outputs)

        # for each convolutional block of the encoder part with separate
        # branches:
        for _ in range(n_encoder_blocks_with_separate_branches):
            current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

            # processing via convolutions:
            for _ in range(n_convolutional_layers_per_block):
                enriched_convolutional_layer = (
                    build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                )
                current_ppg_representation = enriched_convolutional_layer(
                    current_ppg_representation
                )
                enriched_convolutional_layer = (
                    build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                )
                current_ecg_representation = enriched_convolutional_layer(
                    current_ecg_representation
                )

            # downsampling:
            pooling_layer = MaxPooling1D(
                pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                padding=PADDING_MODE
            )
            current_ppg_representation = pooling_layer(
                current_ppg_representation
            )
            pooling_layer = MaxPooling1D(
                pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                padding=PADDING_MODE
            )
            current_ecg_representation = pooling_layer(
                current_ecg_representation
            )

        # --------------------------------------------------------------------
        # ENCODER - FUSED BRANCHES
        # --------------------------------------------------------------------

        current_outputs = Concatenate(axis=-1)(
            [current_ppg_representation, current_ecg_representation]
        )

        # for each convolutional block of the encoder part fusing branches:
        for _ in range(n_encoder_blocks_fusing_branches):
            current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

            # processing via convolutions:
            for _ in range(n_convolutional_layers_per_block):
                enriched_convolutional_layer = (
                    build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                )
                current_outputs = enriched_convolutional_layer(
                    current_outputs
                )

            # downsampling:
            pooling_layer = MaxPooling1D(
                pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                padding=PADDING_MODE
            )
            current_outputs = pooling_layer(current_outputs)

    else:
        # --------------------------------------------------------------------
        # ENCODER WITH END-TO-END UNIFIED BRANCHES
        # --------------------------------------------------------------------

        # for each convolutional block of the encoder:
        for _ in range(n_total_encoder_blocks):
            current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

            # processing via convolutions:
            for _ in range(n_convolutional_layers_per_block):
                enriched_convolutional_layer = (
                    build_enriched_convolutional_layer(
                        convolution_dimension=1,
                        n_filters=current_n_convolutional_filters
                    )
                )
                current_outputs = enriched_convolutional_layer(
                    current_outputs
                )

            # downsampling:
            pooling_layer = MaxPooling1D(
                pool_size=DOWN_AND_UP_SAMPLING_SPAN,
                padding=PADDING_MODE
            )
            current_outputs = pooling_layer(current_outputs)

    # ------------------------------------------------------------------------
    # BOTTLENECK
    # ------------------------------------------------------------------------

    current_n_convolutional_filters *= N_FILTERS_MULTIPLICATION_FACTOR

    # processing via convolutions:
    for _ in range(n_convolutional_layers_per_block):
        enriched_convolutional_layer = build_enriched_convolutional_layer(
            convolution_dimension=1,
            n_filters=current_n_convolutional_filters
        )
        current_outputs = enriched_convolutional_layer(current_outputs)

    current_outputs = Flatten()(current_outputs)

    # ------------------------------------------------------------------------
    # DBP OUTPUT BRANCH
    # ------------------------------------------------------------------------

    current_dbp_outputs = current_outputs

    iterations_left = 3
    n_output_neurons = current_n_convolutional_filters
    while iterations_left > 0:
        iterations_left -= 1
        n_output_neurons /= 2

        enriched_fully_connected_layer = build_enriched_fully_connected_layer(
            # only two output scalars are predicted for signal-wise DBP and
            # SBP regression:
            n_outputs=n_output_neurons if iterations_left > 0 else 1,
            layer_name=None if iterations_left > 0 else OUTPUT_NAME_FOR_DBP
        )
        current_dbp_outputs = enriched_fully_connected_layer(
            current_dbp_outputs
        )

    # ------------------------------------------------------------------------
    # SBP OUTPUT BRANCH
    # ------------------------------------------------------------------------

    current_sbp_outputs = current_outputs

    iterations_left = 3
    n_output_neurons = current_n_convolutional_filters
    while iterations_left > 0:
        iterations_left -= 1
        n_output_neurons /= 2

        enriched_fully_connected_layer = build_enriched_fully_connected_layer(
            # only two output scalars are predicted for signal-wise DBP and
            # SBP regression:
            n_outputs=n_output_neurons if iterations_left > 0 else 1,
            layer_name=None if iterations_left > 0 else OUTPUT_NAME_FOR_SBP
        )
        current_sbp_outputs = enriched_fully_connected_layer(
            current_sbp_outputs
        )

    # ------------------------------------------------------------------------

    return Model(
        inputs=[signals],
        outputs={
            OUTPUT_NAME_FOR_DBP: current_dbp_outputs,
            OUTPUT_NAME_FOR_SBP: current_sbp_outputs
        }
    )


def build_2d_fcn_encoder_decoder_architecture(  # pylint: disable=unused-argument
        n_convolutional_layers_per_block: int,
        n_encoder_blocks: int,
        n_filters_in_first_layer: int,
        **unrelevant_kwargs: Any
) -> Type[Model]:
    """
    Return an instance of a fully-convolutional network (FCN) encoder-decoder
    model architecture for signal prediction from 2-dimensional input signals
    representing spectrograms.
    """
    signals = Input(
        shape=(N_SPECTROGRAM_FREQUENCIES, N_SPECTROGRAM_SAMPLES, 2),
        dtype=DATA_TYPE_FOR_INPUTS
    )

    current_outputs = signals

    # ------------------------------------------------------------------------
    # ENCODER
    # ------------------------------------------------------------------------

    # processing via convolutions to reduce the frequency information into
    # feature channels of a 1-dimensional temporal signal:
    for block_index in range(n_convolutional_layers_per_block):
        enriched_convolutional_layer = build_enriched_convolutional_layer(
            convolution_dimension=2,
            n_filters=n_filters_in_first_layer,
            kernel_size_on_frequency_axis=(
                N_SPECTROGRAM_FREQUENCIES if block_index == 0 else 1
            )
        )
        current_outputs = enriched_convolutional_layer(current_outputs)

    current_outputs = Lambda(
        function=lambda outputs: squeeze(input=outputs, axis=1)
    )(current_outputs)

    # ------------------------------------------------------------------------
    # BOTTLENECK
    # ------------------------------------------------------------------------

    current_n_convolutional_filters = (
        n_filters_in_first_layer / N_FILTERS_MULTIPLICATION_FACTOR
    )

    # processing via convolutions:
    for _ in range(n_convolutional_layers_per_block):
        enriched_convolutional_layer = build_enriched_convolutional_layer(
            convolution_dimension=1,
            n_filters=current_n_convolutional_filters
        )
        current_outputs = enriched_convolutional_layer(current_outputs)

    # ------------------------------------------------------------------------
    # DECODER
    # ------------------------------------------------------------------------

    n_decoder_blocks = round(
        number=log(
            (
                N_UCI_SIGNAL_SEGMENT_SAMPLES / N_SPECTROGRAM_SAMPLES
            ),  # argument of the logarithm
            DOWN_AND_UP_SAMPLING_SPAN  # base of the logarithm
        )
    )

    # for each convolutional block of the decoder:
    for _ in range(n_decoder_blocks):
        current_n_convolutional_filters /= N_FILTERS_MULTIPLICATION_FACTOR

        # upsampling while processing via transposed convolutions:
        enriched_convolutional_layer = build_enriched_convolutional_layer(
            convolution_dimension=1,
            n_filters=current_n_convolutional_filters,
            transposed=True,
            upsampling_span=DOWN_AND_UP_SAMPLING_SPAN
        )
        current_outputs = enriched_convolutional_layer(current_outputs)

        # processing via convolutions:
        for _ in range(n_convolutional_layers_per_block):
            enriched_convolutional_layer = build_enriched_convolutional_layer(
                convolution_dimension=1,
                n_filters=current_n_convolutional_filters
            )
            current_outputs = enriched_convolutional_layer(current_outputs)

    # ------------------------------------------------------------------------

    current_outputs = Lambda(
        function=lambda outputs: outputs[:, :N_UCI_SIGNAL_SEGMENT_SAMPLES, :]
    )(current_outputs)

    current_outputs = Convolution1D(
        # a single output scalar is predicted at any timestep for punctual
        # regression:
        filters=1,
        **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
        **REGULARIZATION_KWARGS
    )(current_outputs)

    return Model(
        inputs=[signals],
        outputs=current_outputs
    )


def build_mlp_encoder_cnn_decoder_architecture(  # noqa: E501 pylint: disable=unused-argument
        n_convolutional_layers_per_block: int,
        n_fully_connected_layers: int,
        n_weights_in_first_layer: int,
        n_filters_in_first_decoder_layer: int,
        **unrelevant_kwargs: Any
) -> Type[Model]:
    """
    Retun an instance of a model architecture with an encoder based on a
    multi-layer perceptron (MLP) and a decoder based on a convolutional neural
    network (CNN) for signal prediction from inputs representing Fuorier
    Transforms.
    """
    signals = Input(
        shape=(N_UCI_SIGNAL_SEGMENT_SAMPLES, 4),
        dtype=DATA_TYPE_FOR_INPUTS
    )

    # considering only the positive part of the spectra, since the negative
    # part is symmetric and carries redundant information:
    current_outputs = Lambda(
        function=lambda outputs: outputs[
            :,
            :int(N_UCI_SIGNAL_SEGMENT_SAMPLES/2),
            :
        ]
    )(signals)

    # ------------------------------------------------------------------------
    # FREQUENCY BIN AVERAGING
    # ------------------------------------------------------------------------

    # averaging the information over adjacent frequency bins to lower spectral
    # resolution (in a channel-independent fashion):
    for _ in range(N_FREQUENCY_BIN_AVERAGING_STEPS):
        pooling_layer = AveragePooling1D(
            pool_size=FREQUENCY_BIN_AVERAGING_DOWNSAMPLING_SPAN,
            padding='valid'
        )
        current_outputs = pooling_layer(current_outputs)

    # ------------------------------------------------------------------------
    # ENCODER
    # ------------------------------------------------------------------------

    current_outputs = Flatten()(current_outputs)

    current_n_fully_connected_weights = (
        n_weights_in_first_layer * N_DENSE_WEIGHTS_DROP_FACTOR
    )

    # for each fully-connected layer of the encoder:
    for _ in range(n_fully_connected_layers):
        current_n_fully_connected_weights /= N_DENSE_WEIGHTS_DROP_FACTOR

        # processing via an empowered fully-connected layer:
        enriched_fully_connected_layer = build_enriched_fully_connected_layer(
            n_outputs=current_n_fully_connected_weights
        )
        current_outputs = enriched_fully_connected_layer(current_outputs)

    # ------------------------------------------------------------------------
    # BOTTLENECK
    # ------------------------------------------------------------------------

    # making features become channels of 1-D unitary-length signal
    # representations, so that transposed-convolutions can be meaningfully
    # applied to expand signal length up to the expected 1-D dimension output
    # length:
    current_outputs = Permute(dims=(2, 1))(
        Lambda(function=lambda outputs: expand_dims(input=outputs, axis=-1))(
            current_outputs
        )
    )

    # ------------------------------------------------------------------------
    # DECODER
    # ------------------------------------------------------------------------

    # number of convolutional decoder blocks computed so as to get the input
    # signal block segment lenght as output length as well:
    n_decoder_blocks = round(
        number=log(
            (N_UCI_SIGNAL_SEGMENT_SAMPLES / 1),  # argument of the logarithm
            DOWN_AND_UP_SAMPLING_SPAN  # base of the logarithm
        )
    )

    current_n_convolutional_filters = (
        n_filters_in_first_decoder_layer * N_FILTERS_MULTIPLICATION_FACTOR
    )

    # for each convolutional block of the decoder:
    for _ in range(n_decoder_blocks):
        current_n_convolutional_filters /= N_FILTERS_MULTIPLICATION_FACTOR

        # upsampling while processing via transposed convolutions:
        enriched_convolutional_layer = build_enriched_convolutional_layer(
            convolution_dimension=1,
            n_filters=current_n_convolutional_filters,
            transposed=True,
            upsampling_span=DOWN_AND_UP_SAMPLING_SPAN
        )
        current_outputs = enriched_convolutional_layer(current_outputs)

        # processing via convolutions:
        for _ in range(n_convolutional_layers_per_block):
            enriched_convolutional_layer = build_enriched_convolutional_layer(
                convolution_dimension=1,
                n_filters=current_n_convolutional_filters
            )
            current_outputs = enriched_convolutional_layer(current_outputs)

    # ------------------------------------------------------------------------

    current_outputs = Lambda(
        function=lambda outputs: outputs[:, :N_UCI_SIGNAL_SEGMENT_SAMPLES, :]
    )(current_outputs)

    current_outputs = Convolution1D(
        # a single output scalar is predicted at any timestep for punctual
        # regression:
        filters=1,
        **dict(COMMON_CONVOLUTION_KWARGS, kernel_size=1),
        **REGULARIZATION_KWARGS
    )(current_outputs)

    return Model(
        inputs=[signals],
        outputs=current_outputs
    )


def build_model_architecture(
        data: ValidArgumentsForDataSource,
        domain: ValidArgumentsForInputDomain,
        task: ValidArgumentsForTask,
        **architecture_hyperparameters: Any  # TODO: remove/resuse
) -> Type[Model]:
    """
    Build a different model architecture based on the input specification,
    with respective hyperparameters.
    """
    assert data in get_args(tp=ValidArgumentsForDataSource)
    assert domain in get_args(tp=ValidArgumentsForInputDomain)
    assert task in get_args(tp=ValidArgumentsForTask)

    if data == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE:
        if (
                domain != ARGUMENT_NAME_FOR_DOMAIN_AS_TIME or
                task != ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM
        ):
            raise NotImplementedError(
                f"Only '{ARGUMENT_NAME_FOR_DOMAIN_AS_TIME}' input domain " +
                f"and '{ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM}' prediction " +
                "task are supported as arguments with " +
                f"'{ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE}' data source."
            )

    if domain == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME:
        if task == ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM:
            vanilla_unet = build_1d_fcn_encoder_decoder_architecture(
                in_house_data=(
                    data == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE
                ),
                multi_branch_encoder=VANILLA_UNET_MULTI_BRANCH_ENCODER,
                multi_res_variant=False,
                n_convolutional_layers_per_block=VANILLA_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
                n_encoder_blocks=VANILLA_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS,
                n_filters_in_first_layer=VANILLA_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER,
                recurrent_encoder=VANILLA_UNET_GRU_AFTER_ENCODER,
                recurrent_bottleneck=VANILLA_UNET_GRU_AFTER_BOTTLENECK,
                recurrent_decoder=VANILLA_UNET_GRU_AFTER_DECODER,
                recurrent_skip_connections=VANILLA_UNET_GRU_ALONG_SKIP_CONNECTIONS,
                skip_connections=VANILLA_UNET_SKIP_CONNECTIONS
            )

            if DOUBLE_UNET_VARIANT:
                multi_res_unet = build_1d_fcn_encoder_decoder_architecture(
                    in_house_data=(
                        data == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE
                    ),
                    multi_branch_encoder=False,
                    multi_res_variant=True,
                    n_convolutional_layers_per_block=MULTI_RES_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
                    n_encoder_blocks=MULTI_RES_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS,
                    n_filters_in_first_layer=MULTI_RES_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER,
                    recurrent_encoder=MULTI_RES_UNET_GRU_AFTER_ENCODER,
                    recurrent_bottleneck=MULTI_RES_UNET_GRU_AFTER_BOTTLENECK,
                    recurrent_decoder=MULTI_RES_UNET_GRU_AFTER_DECODER,
                    recurrent_skip_connections=MULTI_RES_UNET_GRU_ALONG_SKIP_CONNECTIONS,
                    skip_connections=MULTI_RES_UNET_SKIP_CONNECTIONS
                )

                return Sequential(layers=[vanilla_unet, multi_res_unet])

            else:
                return vanilla_unet

        else:
            return build_1d_fcn_encoder_architecture(
                in_house_data=(
                    data == ARGUMENT_NAME_FOR_DATA_SOURCE_AS_IN_HOUSE
                ),
                initially_separate_branches=INITIALLY_SEPARATE_BRANCHES_IN_ENCODER_ONLY_MODEL,
                n_convolutional_layers_per_block=VANILLA_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
                n_total_encoder_blocks=VANILLA_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS,
                n_filters_in_first_layer=VANILLA_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER
            )

    elif domain == ARGUMENT_NAME_FOR_DOMAIN_AS_FREQUENCY:
        if task == ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM:
            return build_mlp_encoder_cnn_decoder_architecture(
                n_convolutional_layers_per_block=VANILLA_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
                n_fully_connected_layers=N_FULLY_CONNECTED_LAYERS,
                n_weights_in_first_layer=N_WEIGHTS_IN_FIRST_ENCODER_LAYER,
                n_filters_in_first_decoder_layer=N_FILTERS_IN_FIRST_DECODER_LAYER
            )

        else:
            raise NotImplementedError

    elif domain == ARGUMENT_NAME_FOR_DOMAIN_AS_TIME_FREQUENCY:
        if task == ARGUMENT_NAME_FOR_TASK_AS_WAVEFORM:
            return build_2d_fcn_encoder_decoder_architecture(
                n_convolutional_layers_per_block=VANILLA_UNET_N_CONVOLUTIONAL_LAYERS_PER_BLOCK,
                n_encoder_blocks=VANILLA_UNET_N_CONVOLUTIONAL_ENCODER_BLOCKS,
                n_filters_in_first_layer=VANILLA_UNET_N_FILTERS_IN_FIRST_ENCODER_LAYER
            )

        else:
            raise NotImplementedError

    else:
        raise Exception("Ill-conceived code.")


if __name__ == '__main__':
    input_domain, prediction_task, data_source, _ = get_required_arguments(
        input_domain=True,
        task=True,
        data_source=True,
        experiment_name=False
    )

    model = build_model_architecture(
        data=data_source,
        domain=input_domain,
        task=prediction_task
    )

    model.summary()
    plot_model(
        model,
        to_file=path_join(MODELING_DIR_NAME, 'model.png'),
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True
    )

else:
    from samples_and_labels import EXPERIMENT_NAME

    EXPERIMENT_DIR = path_join(
        ROOT_DIR,
        RESULTS_DIR_NAME,
        EXPERIMENT_NAME
    )
    MODEL_FILE_PATH = path_join(
        EXPERIMENT_DIR,
        'trained_model.h5'
    )
