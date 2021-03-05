import tensorflow as tf
import tensorflow.keras.layers as layers

from datetime import datetime

import dataloader_ae as DataLoader

data_train = DataLoader.LoaderAE(path_data='../data/dl_vdj_train.csv', batch_size=32).training_data
data_val = DataLoader.LoaderAE(path_data='../data/dl_vdj_val.csv', batch_size=32).training_data

loss = tf.keras.losses.MSE


early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
SAVE_MODEL = True

shape = (25,)


def get_auto_encoder(encoder, decoder, input_shape):
    tcr = layers.Input(shape=input_shape)
    latent = encoder(tcr)
    reconstructed = decoder(latent)
    return tf.keras.models.Model(inputs=tcr, outputs=reconstructed)


def get_encoder(latent_dim, input_shape):
    encoder = tf.keras.Sequential()
    architecture = [
        layers.Embedding(21, 10, input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(latent_dim),
        layers.Lambda(l2_norm)
    ]
    for layer in architecture:
        encoder.add(layer)
    return encoder


def l2_norm(x):
    return tf.math.l2_normalize(x, axis=-1)


def get_decoder(output_size):
    decoder = tf.keras.Sequential()
    architecture = [
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(300, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(output_size[0]*output_size[1]),
        layers.Reshape(output_size),
        layers.Softmax()
    ]
    for layer in architecture:
        decoder.add(layer)
    return decoder


latent = 32
encoder_model = get_encoder(latent, shape)
decoder_model = get_decoder((25, 21))
model = get_auto_encoder(encoder_model, decoder_model, shape)


print(model.summary())
log_dir = f'logs_ae/test' + datetime.now().strftime('%m%d%Y_%H%M%S')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))
model.fit(data_train, validation_data=data_val, epochs=100, verbose=1, callbacks=[early_stopping, tensorboard])

if SAVE_MODEL:
    tf.keras.models.save_model(encoder_model, '../trained_models/test_ae_model', save_format='h5')
