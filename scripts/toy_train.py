import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.layers as layers

import numpy as np

import Losses
import MetricsTF


class DataLoader:
    def __init__(self, mode, classes_per_batch, samples_per_class):
        if mode == 'train':
            (self.data, self.labels), _ = mnist.load_data()
        else:
            _, (self.data, self.labels) = mnist.load_data()

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class

        self.data_by_label = self.transform_data()

    def transform_data(self):
        samples = {}
        for img, target in zip(self.data, self.labels):
            if target not in samples:
                samples[target] = []
            samples[target].append(np.expand_dims(img, -1))
        return samples

    def sample_generator(self):
        while True:
            classes = np.random.choice(list(self.data_by_label.keys()), size=self.classes_per_batch, replace=False)
            for cls in classes:
                samples = np.random.choice(range(len(self.data_by_label[cls])), size=self.samples_per_class,
                                           replace=False)
                for idx in samples:
                    yield self.data_by_label[cls][idx], np.array(int(cls))

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.sample_generator, (tf.float32, tf.int16))
        dataset = dataset.batch(self.classes_per_batch * self.samples_per_class)
        return dataset


def get_model():
    simple_model = tf.keras.Sequential()
    architecture = [
        layers.Conv2D(24, kernel_size=5, padding="same", activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.3),
        layers.Conv2D(48, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        # layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(48),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        ]
    for layer in architecture:
        simple_model.add(layer)
    return simple_model


model = get_model()
data_train = DataLoader('train', 8, 2).get_dataset()
data_test = DataLoader('test', 8, 2).get_dataset()

# loss = Losses.contrastive_loss_at_margin(margin_neg=1, margin_pos=0.0, weight_pos=1.)
loss = Losses.contrastive_loss_tfa

metrics = [
    MetricsTF.recall_at_k(1),
    MetricsTF.recall_at_k(4),
]
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), metrics=metrics)

history = model.fit(data_train, epochs=1000, validation_data=data_test,
                    steps_per_epoch=500, validation_steps=100,
                    verbose=1, callbacks=[early_stopping])

# for batch, labels in data_train:
#     print(batch)
#     break