import tensorflow as tf
from datetime import datetime

from PairedDataModels.DataLoaderPaired import DataLoader
import Models

training_data = DataLoader(128, path_dataset='../data/dl_iedb_train.csv').get_dataset()
validation_data = DataLoader(128, path_dataset='../data/dl_iedb_val.csv').get_dataset()

input_shape = (2, 25)  # , 21)


# model = Models.perceptron_test(input_shape)
# body = Models.body_lstm()
# body = Models.body_bi_lstm()
# body = Models.body_fcl()
# head = Models.head_cosine()
# head = Models.head_euclidean()
# model = Models.inceptionish_siamese(input_shape)
# model = Models.cnn_lstm_attention(input_shape)


loss = Models.bce_loss()
# loss = Models.contrastive_loss(0.2)

# cb_mini_val = InterEpochValidation.InterEpochValidator(3000)
# metric_mean = CustomKeras.metric_mean
# metric_std = MetricsTF.metric_std

metrics = [
    tf.keras.metrics.BinaryAccuracy(threshold=0.5)
]

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

log_dir = f'logs_tests/bilstm' + datetime.now().strftime('%m%d%Y_%H%M%S')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print(log_dir)
body = Models.body_bi_lstm()
head = Models.head_fcl('sigmoid', do_concat=False)
model = Models.general_siamese(input_shape, siamese_body=body, siamese_head=head)

model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              metrics=metrics)

history = model.fit(training_data, epochs=100, validation_data=validation_data, verbose=1,
                    callbacks=[tensorboard, early_stopping])

