import numpy as np
import tensorflow as tf
from datetime import datetime

import DataLoader as DataLoader
import Models as Models
import Losses

import MetricsTF


training_data = DataLoader.BatchSampler(16, 2, 'data/distanceLearning_train.csv').get_dataset()
validation_data = DataLoader.BatchSampler(16, 2, 'data/distanceLearning_val.csv').get_dataset()


loss = Losses.contrastive_loss_at_margin(1.)

metrics = [
    MetricsTF.recall_at_k(1),
    MetricsTF.recall_at_k(4),
    MetricsTF.recall_at_k(8),
    MetricsTF.recall_at_k(32)
]


early_stopping = tf.keras.callbacks.EarlyStopping(patience=20)

log_dir = f'logs_sampler/bilstm' + datetime.now().strftime('%m%d%Y_%H%M%S')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print(log_dir)


model = Models.body_bi_lstm()
print(model.summary())

model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), metrics=metrics)

history = model.fit(training_data, validation_data=validation_data, epochs=100,
                    steps_per_epoch=500, validation_steps=100,
                    verbose=1, callbacks=[early_stopping, tensorboard])


