# model = Models.perceptron_test(input_shape)
# body = Models.body_lstm()
# body = Models.body_bi_lstm()
# body = Models.body_fcl()
# head = Models.head_cosine()
# head = Models.head_euclidean()
# model = Models.inceptionish_siamese(input_shape)
# model = Models.cnn_lstm_attention(input_shape)


'''
for layers_fc in [0]:
    for fc_size in [128]:
        for l2 in [0., 0.001, 0.0001]:
            for csize in [5, 7]:
                for fil in [32, 64]:
                    for layers_conv in [1, 2, 3, 4, 5]:
                        for do_norm in [False]:
                            if layers_fc == 0 and fc_size != 128:
                                continue
                            body = Models.body_cnn(amount_convs=layers_conv, filters=fil, size_conv=csize,
                                                   amount_fc=layers_fc, size_fc=layers_fc, l2_reg=l2,
                                                   do_normalize=do_norm)
                            head = Models.head_fcl('sigmoid', do_concat=False)
                            model = Models.general_siamese(input_shape, siamese_body=body, siamese_head=head)

                            log_dir = f'logs_hs2/hs_conv{layers_conv}_fil{fil}_csize{csize}_fc{layers_fc}'
                            log_dir += f'_fcsize{fc_size}_l2{l2}_normal{do_norm}'
                            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                            print(log_dir)

                            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                                          metrics=['binary_accuracy', metric_std])

                            history = model.fit(training_data, epochs=30, validation_data=validation_data, verbose=1,
                                                callbacks=[tensorboard, early_stopping])
'''

'''
for do in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for fc_size in [128]:
        for fc_layers in [0]:
            for lstm_size in [16, 32, 64, 128, 256, 512]:
                for lstm_layers in [1, 2, 3]:
                    if fc_layers == 0 and fc_size != 128:
                        continue
                    log_dir = f'logs_hs3/lstm_layers{lstm_layers}_size{lstm_size}_fc{fc_layers}_fcsize{fc_size}_do{do}'
                    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                    print(log_dir)
                    body = Models.body_lstm(lstm_layers, lstm_size, fc_layers, fc_size, dropout=do, l2_reg=0.)
                    head = Models.head_fcl('sigmoid', do_concat=False)
                    model = Models.general_siamese(input_shape, siamese_body=body, siamese_head=head)
                    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                                  metrics=['binary_accuracy', metric_std])

                    history = model.fit(training_data, epochs=100, validation_data=validation_data, verbose=1,
                                        callbacks=[tensorboard, early_stopping])'''

# really ugly evaluation based on best threshold
'''
pred_val = []
true_val = []

for batch, target in validation_data:
    true_val.append(target.numpy())
    pred_val.append(model.predict(batch))

true_val = np.concatenate(true_val)
pred_val = np.concatenate(pred_val)
pred_val = np.ndarray.flatten(pred_val)
threshs = np.sort(pred_val, axis=0)

max_acc = 0
n = true_val.shape[0]
for thresh in threshs:
    pred_bin = (pred_val > thresh).astype(np.int_)
    trues = np.equal(pred_bin, true_val).sum()
    acc = trues / n
    max_acc = max(max_acc, acc)

print(max_acc)
'''

'''
class InterEpochValidator(tf.keras.callbacks.Callback):
    def __init__(self, steps_before_validation):
        self.steps_before_validation = steps_before_validation
        self.data = DataLoader.get_data(256, 'val', fraction=0.01)
        super().__init__()

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.steps_before_validation == self.steps_before_validation-1:
            self.run_mini_validation()

    def run_mini_validation(self):
        print('        - mini_validation: ', end='')
        self.model.evaluate(self.data, verbose=2)


def metric_mean(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)


def metric_std(y_true, y_pred):
    return tf.keras.backend.std(y_pred)
'''

'''
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# tf.config.run_functions_eagerly(True)
'''