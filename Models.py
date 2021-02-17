import tensorflow as tf
import tensorflow.keras.layers as layers


# todo check whether needed and move
# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)


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
def body_bi_lstm(embedding_size=10, lstm_layers=2, lstm_hidden=100, lstm_dropout=0., l2_reg=0.0001,
                 fc_layers=None, fc_dropout=0.):
    l2_reg = tf.keras.regularizers.L2(l2_reg)

    architecture = [layers.Input(shape=(25, 21))]
    if embedding_size is not None:
        architecture = [layers.Embedding(21, embedding_size, input_shape=(25,), mask_zero=True)]

    for _ in range(lstm_layers-1):
        architecture.append(layers.Bidirectional(layers.LSTM(lstm_hidden, return_sequences=True,
                                                             kernel_regularizer=l2_reg)))
        architecture.append(layers.Dropout(lstm_dropout))
    architecture.append(layers.Bidirectional(layers.LSTM(lstm_hidden, return_sequences=False,
                                                         kernel_regularizer=l2_reg)))

    for idx, hidden_units in enumerate(fc_layers):
        dropout = 0
        activation = 'linear'
        if idx != len(fc_layers)-1:
            activation = 'relu'
            dropout = fc_dropout
        architecture.append(layers.Dense(hidden_units, kernel_regularizer=l2_reg, activation=activation))
        architecture.append(layers.Dropout(dropout))

    architecture.append(layers.Lambda(l2_normalize))
    return architecture


@to_sequential
def body_fcl(fc_layers, embedding_size=10, fc_l2=0, fc_dropout=0., use_bn=False):
    architecture = [layers.Input(shape=(25, 21))]
    if embedding_size is not None:
        architecture = [layers.Embedding(21, embedding_size, input_shape=(25,)),
                        layers.Flatten()]

    for idx, size in enumerate(fc_layers):
        activation = 'relu'
        if idx == len(fc_layers)-1:
            fc_dropout = 0
            activation = 'linear'
        architecture.append(dense_block(size, activation=activation, use_batchnorm=use_bn, dropout=fc_dropout,
                                        l2_reg=fc_l2))
    architecture.append(layers.Lambda(l2_normalize))
    return architecture


@to_sequential
def body_cnn(embedding_size=10, amount_convs=2, filters=32, size_conv=3, amount_fc=1, size_fc=100, l2_reg=0.000):
    architecture = [layers.Input(shape=(25, 21))]
    if embedding_size is not None:
        architecture = [layers.Embedding(21, embedding_size, input_shape=(25,))]

    for _ in range(amount_convs):
        architecture.append(conv_block(filters, size_conv, 1, 'relu', use_batchnorm=False, l2_reg=l2_reg))
    architecture.append(layers.Flatten())

    for _ in range(amount_fc-1):
        architecture.append(dense_block(size_fc, 'relu', l2_reg=l2_reg))
    architecture.append(dense_block(size_fc, 'linear', l2_reg=l2_reg))
    architecture.append(layers.Lambda(l2_normalize))
    return architecture


@to_sequential
def body_cnn_lstm():
    # todo
    raise NotImplementedError('Combination of CNN and LSTM still needs to be implemented.')


@to_sequential
def vgg_like():
    architecture = [
        layers.Input(shape=(25, 21)),
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=2),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=2),
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=2),
        # layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'),
        # layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'),
        # layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'),
        # layers.MaxPool1D(pool_size=2, strides=2),
        # layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'),
        # layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'),
        # layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'),
        # layers.MaxPool1D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(units=4096, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(units=4096, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(units=64, activation='linear'),
        layers.Lambda(l2_normalize),
    ]
    return architecture


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


def l2_normalize(x):
    import tensorflow as tf
    x = tf.math.l2_normalize(x, axis=-1)
    return x
