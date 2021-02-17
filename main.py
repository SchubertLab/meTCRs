import tensorflow as tf
from datetime import datetime

import DataLoader as DataLoader

import Models as Models
import Losses
import TripletLoss

import MetricsTF


# Losses.contrastive_loss_tfa,
# 'semi_hard': Losses.semi_hard_triplet_loss(),

metrics = [
    MetricsTF.recall_at_k(1),
    MetricsTF.recall_at_k(4),
]


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_R@1', mode='max', patience=50, restore_best_weights=True)
SAVE_MODEL = True

models = {
    # 'FC': Models.body_fcl(fc_layers=[8, 8, 8], use_bn=False),
    # 'CNN': Models.body_cnn(amount_convs=1, size_conv=5, filters=16, amount_fc=2, size_fc=8),
    # 'BiLSTM': Models.body_bi_lstm(embedding_size=10, lstm_layers=3, lstm_hidden=500, lstm_dropout=0., l2_reg=0.0000,
    #                 fc_layers=[512, 128], fc_dropout=0.),
    # 'VGG16': Models.vgg_like(),
}


trained_model = None

for positives in [16, 32, 4, 8]:
    training_data = DataLoader.BatchSampler(64, positives, 'data/dl_tcrmatch_train.csv', do_embed=True, do_weight=False).get_dataset()
    validation_data = DataLoader.BatchSampler(32, 2, 'data/dl_tcrmatch_val.csv', do_embed=True).get_dataset()
    for margin in [0.05, 0.1, 0.3, 0.5, 1.0]:
        model = Models.body_bi_lstm(embedding_size=10, lstm_layers=1, lstm_hidden=100, lstm_dropout=0., l2_reg=0.0000,
                    fc_layers=[256, 128], fc_dropout=0.)
        print(model.summary())

        loss = TripletLoss.triplet_loss(margin)

        log_dir = f'logs_hs_large/biLSTM_{margin}_{positives}' + datetime.now().strftime('%m%d%Y_%H%M%S')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), metrics=metrics)

        history = model.fit(training_data, epochs=1000, validation_data=validation_data,
                            steps_per_epoch=500, validation_steps=100,
                            verbose=1, callbacks=[early_stopping, tensorboard])

    # if SAVE_MODEL:
    #     tf.keras.models.save_model(model, 'trained_models/test_model', save_format='h5')
    # trained_model = model


# for batch, label in validation_data:
#     embeddings = model.predict(batch)
#     distances = Compute.pairwise_distance(embeddings)
#     np.set_printoptions(threshold=sys.maxsize)
#     print(embeddings)
#     print(distances)
#     break