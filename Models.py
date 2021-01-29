import tensorflow as tf
import tensorflow.keras.layers as layers


# Util function to create an Tensorflow sequential
def to_sequential(function):
    def create_sequential(*args, **kwargs):
        layer_list = function(*args, **kwargs)
        sequential_model = tf.keras.Sequential()
        for layer in layer_list:
            sequential_model.add(layer)
        return sequential_model
    return create_sequential


# <- Different Model Architectures used for Feature Embedding ->
@to_sequential
def body_bi_lstm():
    l2_reg = tf.keras.regularizers.L2(0.0001)
    architecture = [
        layers.Embedding(21, 10, input_shape=(23,)),
        layers.Bidirectional(layers.LSTM(100, return_sequences=True, kernel_regularizer=l2_reg)),
        # layers.Dropout(0.1),
        layers.Bidirectional(layers.LSTM(100, kernel_regularizer=l2_reg)),
        # layers.Dense(512, kernel_regularizer=l2_reg),
        layers.LayerNormalization(),
    ]
    return architecture


@to_sequential
def body_fcl():
    architecture = [
        layers.Flatten(),
        dense_block(20, 'relu', use_batchnorm=False),
        dense_block(20, 'relu', use_batchnorm=False),
    ]
    return architecture


@to_sequential
def body_cnn(amount_convs, filters, size_conv, amount_fc, size_fc, l2_reg, do_normalize=False):
    architecture = []
    for _ in range(amount_convs):
        architecture.append(conv_block(filters, size_conv, 1, 'relu', use_batchnorm=False, l2_reg=l2_reg))
    architecture.append(layers.Flatten())
    for _ in range(amount_fc):
        architecture.append(dense_block(size_fc, 'relu', l2_reg=l2_reg))
    if do_normalize:
        architecture.append(layers.LayerNormalization())
    return architecture


@to_sequential
def body_lstm(hidden_layers, hidden_size, fc_layers, fc_size, dropout=0., l2_reg=0.):
    architecture = []
    l2 = tf.keras.regularizers.L2(l2_reg)
    for idx in range(hidden_layers):
        return_sequence = idx != hidden_layers-1
        architecture.append(layers.LSTM(hidden_size, kernel_regularizer=l2, return_sequences=return_sequence))
        architecture.append(layers.Dropout(dropout))
    for idx in range(fc_layers):
        architecture.append(layers.Dense(fc_size, kernel_regularizer=l2))
        if idx != len(fc_layers)-1:
            architecture.append(layers.Dropout(dropout))
    return architecture


@to_sequential
def body_cnn_lstm():
    # todo
    raise NotImplementedError('Combination of CNN and LSTM still needs to be implemented.')


# Layers Functions
def conv_block(filters, size, stride, activation, use_batchnorm, l2_reg=0.):
    l2_reg = tf.keras.regularizers.L2(l2_reg)
    block = tf.keras.Sequential()
    block.add(layers.Conv1D(filters, size, strides=stride, padding='same', kernel_regularizer=l2_reg))
    if use_batchnorm:
        block.add(layers.BatchNormalization())
    block.add(layers.Activation(activation))
    return block


def dense_block(size, activation, use_batchnorm=False, dropout=0.0, l2_reg=0.0):
    l2_reg = tf.keras.regularizers.L2(l2_reg)
    block = tf.keras.Sequential()
    block.add(layers.Dense(size, kernel_regularizer=l2_reg))
    if use_batchnorm:
        block.add(layers.BatchNormalization())
    if dropout != 0:
        block.add(layers.Dropout(dropout))
    block.add(layers.Activation(activation))
    return block
