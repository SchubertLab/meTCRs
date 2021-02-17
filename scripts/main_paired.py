import numpy as np
import tensorflow as tf
from datetime import datetime

import DataLoader as DataLoader

import old_code.Models as Models
import Losses

import MetricsTF


training_data = DataLoader.BatchSampler(32, 2, 'data/full_train.csv', do_weight=False).get_dataset(do_paired=True)
validation_data = DataLoader.BatchSampler(32, 2, 'data/full_val.csv').get_dataset(do_paired=True)


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


metrics = [
    'binary_accuracy',
]


early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
SAVE_MODEL = True

# body = Models.body_cnn(amount_convs=2, filters=32, size_conv=3, amount_fc=0, size_fc=0, l2_reg=0, do_normalize=False)
body = Models.body_bi_lstm()
head = Models.head_fcl(final_activation='linear', do_concat=True)
model = Models.general_siamese(input_shape=(2, 25), siamese_body=body, siamese_head=head)

print(model.summary())
log_dir = f'logs_sampler_bin/tester' + datetime.now().strftime('%m%d%Y_%H%M%S')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)

history = model.fit(training_data, validation_data=validation_data, epochs=1000,
                    steps_per_epoch=500, validation_steps=100,
                    verbose=1, callbacks=[early_stopping, tensorboard])

if SAVE_MODEL:
    tf.keras.models.save_model(model, 'trained_models/test_model_bin', save_format='h5')
