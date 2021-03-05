import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as keras


def perceptron_test(input_shape):
    tcrs = layers.Input(shape=input_shape)
    x = layers.Flatten()(tcrs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=tcrs, outputs=x)


def general_siamese(input_shape, siamese_body, siamese_head):
    tcrs = layers.Input(shape=input_shape)
    tcr1 = layers.Lambda(lambda y: y[:, 0])(tcrs)
    tcr2 = layers.Lambda(lambda y: y[:, 1])(tcrs)
    # embedding = layers.Embedding(21, 10, mask_zero=True)

    # tcr1 = embedding(tcr1)
    # tcr2 = embedding(tcr2)

    x1 = siamese_body(tcr1)
    x2 = siamese_body(tcr2)

    x = siamese_head([x1, x2])
    return tf.keras.models.Model(inputs=tcrs, outputs=x)


def inceptionish_siamese(input_shape):
    tcrs = layers.Input(shape=input_shape)
    tcr1 = layers.Lambda(lambda y: y[:, 0, :])(tcrs)
    tcr2 = layers.Lambda(lambda y: y[:, 1, :])(tcrs)

    filters = 5
    conv1 = layers.Conv1D(filters, 1,  activation='relu', padding='same')
    conv3 = layers.Conv1D(filters, 3,  activation='relu', padding='same')
    conv5 = layers.Conv1D(filters, 5,  activation='relu', padding='same')
    conv7 = layers.Conv1D(filters, 7,  activation='relu', padding='same')

    x1_1 = conv1(tcr1)
    x1_3 = conv3(tcr1)
    x1_5 = conv5(tcr1)
    x1_7 = conv7(tcr1)

    x2_1 = conv1(tcr2)
    x2_3 = conv3(tcr2)
    x2_5 = conv5(tcr2)
    x2_7 = conv7(tcr2)

    x1 = layers.concatenate([x1_1, x1_3, x1_5, x1_7])
    x2 = layers.concatenate([x2_1, x2_3, x2_5, x2_7])

    x = layers.concatenate([x1, x2])
    x = layers.Flatten()(x)
    x = layers.Dense(1, 'sigmoid')(x)
    return tf.keras.models.Model(inputs=tcrs, outputs=x)


# Util function to create an Tensorflow sequential
def to_sequential(function):
    def create_sequential(*args, **kwargs):
        layer_list = function(*args, **kwargs)
        sequential_model = tf.keras.Sequential()
        for layer in layer_list:
            sequential_model.add(layer)
        return sequential_model
    return create_sequential


# Siamese Body Architectures
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
    architecture = []  # layers.Input(shape=(25, 21))]
    for _ in range(amount_convs):
        architecture.append(conv_block(filters, size_conv, 1, 'relu', use_batchnorm=False, l2_reg=l2_reg))
    architecture.append(layers.Flatten())
    for _ in range(amount_fc):
        architecture.append(dense_block(size_fc, 'relu', l2_reg=l2_reg))
    if do_normalize:
        architecture.append(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)))
    return architecture


@to_sequential
def body_bi_lstm():
    l2_reg = tf.keras.regularizers.L2(0.0001)
    architecture = [
        layers.Embedding(21, 20, input_shape=(25,), mask_zero=True),
        layers.Bidirectional(layers.LSTM(100, return_sequences=True, kernel_regularizer=l2_reg)),
        # layers.Dropout(0.1),
        layers.Bidirectional(layers.LSTM(100, kernel_regularizer=l2_reg)),
        # layers.Dense(512, kernel_regularizer=l2_reg),
    ]
    return architecture


@to_sequential
def body_cnn_lstm():
    # TODO
    # cnn filter sizes 3 + 5 concat stride 1 + 10 Filter
    # cnn size 3 + 20 Filter
    # bi lstm mit je 10 full seq
    # attention 20 units, aln num 10
    # dense 20
    '''
    Convolutional layer size 3: (128, 10, 400)
    Convolutional layer size 5: (128, 10, 400)
    Concatenated convolutional layers: (128, 20, 400)
    Final convolutional layer: (128, 20, 400)
    Second DimshuffleLayer layer: (128, 400, 20)
    Forward LSTM layer: (128, 400, 15)
    Backward LSTM layer: (128, 400, 15)
    Concatenated hidden states: (128, 400, 30)
    Attention layer: (128, 2, 30)
    Last decoding step: (128, 30)
    Dense layer: (128, 30)
    Output layer: (128, 10)
    '''

    raise NotImplementedError('')


def cnn_lstm_attention(input_shape):
    tcrs = layers.Input(shape=input_shape)
    tcr1 = layers.Lambda(lambda y: y[:, 0, :])(tcrs)
    tcr2 = layers.Lambda(lambda y: y[:, 1, :])(tcrs)

    n_hid = 100

    embedding = layers.Embedding(21, 10)
    conv1_3 = layers.Conv1D(n_hid, 3, activation='relu', padding='same')
    conv1_5 = layers.Conv1D(n_hid, 5, activation='relu', padding='same')
    # concat
    concat1 = layers.Concatenate()
    conv2_3 = layers.Conv1D(2*n_hid, 3, activation='relu', padding='same')

    lstm = layers.Bidirectional(layers.LSTM(n_hid))
    # non line tanh?
    # att = layers.Attention()

    # concat?
    dense1 = layers.Dense(3*n_hid, activation='relu')
    # drop_1 = layers.Dropout(0.5)
    dense2 = layers.Dense(1, activation='sigmoid')

    def feature_extraction(tcr):
        tcr = embedding(tcr)
        x1_3 = conv1_3(tcr)
        x1_5 = conv1_5(tcr)
        x_ = concat1([x1_3, x1_5])
        x_ = conv2_3(x_)
        x_ = lstm(x_)
        # x_ = att(x_)
        x_ = layers.Flatten()(x_)
        x_ = layers.LayerNormalization()(x_)
        return x_

    x1 = feature_extraction(tcr1)
    x2 = feature_extraction(tcr2)

    x = layers.subtract([x1, x2])
    x = layers.Lambda(lambda y: tf.keras.backend.abs(y))(x)

    # x = layers.Lambda(lambda y: y ** 2)(x)
    # x = layers.Lambda(lambda y: tf.reduce_sum(y, axis=-1, keepdims=True))(x)
    x = dense1(x)
    # x = drop_1(x)
    x = dense2(x)
    return tf.keras.models.Model(inputs=tcrs, outputs=x)


# Siamese Head Architectures

@to_sequential
def head_euclidean():
    architecture = [
        layers.Subtract(),
        layers.Lambda(lambda y: y ** 2),
        layers.Lambda(lambda y: tf.reduce_sum(y, axis=-1, keepdims=True))
    ]
    return architecture


@to_sequential
def head_cosine():
    architecture = [
        layers.Dot(axes=1, normalize=True),
        layers.Lambda(lambda y: 1-y),
    ]
    return architecture


@to_sequential
def head_fcl(final_activation='sigmoid', do_concat=False):
    if do_concat:
        architecture = [layers.Concatenate()]
    else:
        architecture = [layers.Subtract(),
                        layers.Lambda(lambda y: keras.abs(y))]
    architecture += [
        # layers.Lambda(lambda y: y**2),
        # dense_block(20, 'relu', use_batchnorm=False),
        dense_block(256, 'relu', l2_reg=0.00, dropout=0.0),
        dense_block(1, final_activation, use_batchnorm=False, l2_reg=0.000),
    ]
    return architecture


# loss functions

def contrastive_loss(margin):
    @tf.function
    def loss_fct(target, distance):
        dist_sqrt = keras.sqrt(distance)
        loss_neg = keras.square(keras.maximum(0., margin - dist_sqrt))
        loss_total = target * loss_neg
        loss_total += (1 - target) * distance
        loss_total = keras.mean(loss_total)
        return loss_total

    return loss_fct


def bce_loss(label_smoothing=0.0):
    return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)


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
