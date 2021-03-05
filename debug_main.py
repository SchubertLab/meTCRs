import tensorflow as tf
from datetime import datetime

import DataLoader as DataLoader

import old_code.Models as Models
import Losses
import TripletLoss

import MetricsTF

# Losses.contrastive_loss_tfa,
# 'semi_hard': Losses.semi_hard_triplet_loss(),

metrics = [
    'binary_accuracy',
    # MetricsTF.recall_at_k(1),
    # MetricsTF.recall_at_k(4),
]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='max', patience=50, restore_best_weights=True)
SAVE_MODEL = True

margin = 1
positives = 2
latent_size = 8

training_data = DataLoader.BatchSampler(32, positives, 'data/dl_tcrmatch_train.csv', min_entries_per_group=500,
                                        do_embed=True, do_weight=False, encoding='blosum').get_dataset(do_paired=True)
validation_data = DataLoader.BatchSampler(32, 2, 'data/dl_tcrmatch_val.csv', do_embed=True,
                                          encoding='blosum').get_dataset(do_paired=True)


body = Models.body_cnn(amount_convs=5, filters=32, size_conv=3, amount_fc=2, size_fc=16, l2_reg=0, do_normalize=True)
head = Models.head_fcl()
model = Models.general_siamese(input_shape=(2, 25, 21), siamese_body=body, siamese_head=head)

print(model.summary())

# loss = TripletLoss.triplet_loss(margin)
# loss = Models.contrastive_loss(margin)
loss = Models.bce_loss()
# loss = Losses.semi_hard_triplet_loss()

log_dir = f'logs_debug/cnn_{margin}_{positives}_{latent_size}_'
log_dir += datetime.now().strftime('%m%d%Y_%H%M%S')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), metrics=metrics)


# batch_train = None
# for batch in training_data:
#     batch_train = batch
#     break

history = model.fit(training_data, epochs=1000, validation_data=validation_data,
                    steps_per_epoch=500, validation_steps=100,
                    verbose=1, callbacks=[early_stopping, tensorboard])

# if SAVE_MODEL:
#     tf.keras.models.save_model(model, 'trained_models/test_model', save_format='h5')
# trained_model = model
