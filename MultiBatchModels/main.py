import tensorflow as tf
from datetime import datetime

import Utils.Configurations as Dirs

import MultiBatchModels.Models as Models
import MultiBatchModels.Losses as Losses
import MultiBatchModels.DataLoader as DataLoader

from evaluation import MetricsTF

# Losses.contrastive_loss_tfa,
# 'semi_hard': Losses.semi_hard_triplet_loss(),

DATASET_KEY = 'IEDB'

metrics = [
    MetricsTF.recall_at_k(1),
    MetricsTF.recall_at_k(4),
]


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_R@1', mode='max', patience=10, restore_best_weights=True)
SAVE_MODEL = True

models = {
    # 'FC': Models.body_fcl(fc_layers=[8, 8, 8], use_bn=False),
    # 'CNN': Models.body_cnn(amount_convs=1, size_conv=5, filters=16, amount_fc=2, size_fc=8),
    # 'BiLSTM': Models.body_bi_lstm(embedding_size=10, lstm_layers=3, lstm_hidden=500, lstm_dropout=0., l2_reg=0.0000,
    #                 fc_layers=[512, 128], fc_dropout=0.),
    # 'VGG16': Models.vgg_like(),
}


trained_model = None

for positives in [2,]:
    training_data = DataLoader.BatchSampler(16, positives, Dirs.PATH_DATA_TRAIN[DATASET_KEY],
                                            do_embed=True, encoding='one_hot', do_weight=True).get_dataset()
    validation_data = DataLoader.BatchSampler(16, 2, Dirs.PATH_DATA_VAL[DATASET_KEY],
                                              do_embed=True, encoding='one_hot').get_dataset()
    for margin in [0.6]:
        for latent_size in [8]:
            # model = Models.body_bi_lstm(embedding_size=None, lstm_layers=1, lstm_hidden=50, lstm_dropout=0.0,
            #                             l2_reg=0.0, fc_layers=[256, latent_size], fc_dropout=0.)
            model = Models.body_cnn(amount_convs=5, size_conv=5, filters=16, sizes_fc=[256, latent_size])

            print(model.summary())

            # loss = TripletLoss.triplet_loss(margin)
            loss = Losses.get_contrastive_loss_tfa(margin)

            log_dir = Dirs.PATH_LOGS + f'/hs2_large/lstm_{margin}_{positives}_{latent_size}_'
            log_dir += datetime.now().strftime('%m%d%Y_%H%M%S')
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                          metrics=metrics)

            history = model.fit(training_data, epochs=1, validation_data=validation_data,
                                steps_per_epoch=500, validation_steps=100,
                                verbose=1, callbacks=[early_stopping, tensorboard])
            trained_model = model
if SAVE_MODEL:
    tf.keras.models.save_model(trained_model, Dirs.PATH_MODEL_MULTI_BATCH, save_format='h5')



# for batch, label in validation_data:
#     embeddings = model.predict(batch)
#     distances = Compute.pairwise_distance(embeddings)
#     np.set_printoptions(threshold=sys.maxsize)
#     print(embeddings)
#     print(distances)
#     break