import numpy as np
from bottleneck import argpartition
import tensorflow as tf


def recall_at_k(distance_matrix, labels, k_values):
    recalls = {}
    for k in k_values:
        correct, count = 0., 0.
        for i in range(labels.shape[0]):
            distance_matrix[i, i] = 1e10
            k_neighbors = argpartition(distance_matrix[i], k)[:k]
            if any(labels[i] == labels[neighbor] for neighbor in k_neighbors):
                correct += 1
            count += 1
        recalls[k] = correct/count
    return recalls

'''
def keras_recall_at_1(distance_matrix, labels):
    label_mask = tf.math.equal(labels, tf.transpose(labels),)
    label_mask = tf.cast(label_mask, dtype=tf.dtypes.float32)

    distances_masked = -1. * (distance_matrix + tf.eye(distance_matrix.shape[-1]) * 1e5)
    _, top_k = tf.math.top_k(distances_masked, k=1)
    mask_top_k = tf.one_hot(top_k, depth=distance_matrix.shape[-1])
    mask_top_k = tf.reduce_sum(mask_top_k, axis=-2)
    correct_predictions_at_k = label_mask * mask_top_k
    correct_predictions_at_k = tf.math.reduce_sum(correct_predictions_at_k, axis=-1)
    recall_value = tf.math.count_nonzero(correct_predictions_at_k) / distance_matrix.shape[-1]
    return recall_value
'''

if __name__ == '__main__':
    labels = tf.constant([1., 1., 2., 2., 3., 3.], shape=(1, 6))
    distances = tf.constant([
        [0., 1., 2., 3., 4., 5.],
        [1., 0., 6., 7., 8., 9.],
        [2., 6., 0., 1., 2., 3.],
        [3., 7., 1., 0., 4., 5.],
        [4., 8., 2., 4., 0., 6.],
        [5., 9., 3., 5., 6., 0.]
    ]
    )
    print(keras_recall_at_1(distances, labels))


'''
Gather thoughts:
labels = 
1
1
2
2
3
3

label_mask = 
1 1 0 0 0 0
1 1 0 0 0 0
0 0 1 1 0 0
0 0 1 1 0 0
0 0 0 0 1 1
0 0 0 0 1 1

distances= 
0 1 2 3 4 5
1 0 6 7 8 9
2 6 0 1 2 3
3 7 1 0 4 5
4 8 2 4 0 6
5 9 3 5 6 0

distances_masked =
x 1 2 3 4 5
1 x 6 7 8 9
2 6 x 1 2 3
3 7 1 x 4 5
4 8 2 4 x 6
5 9 3 5 6 x

top 2 =
0 1 1 0 0 0
1 0 1 0 0 0
1 0 0 1 0 0
1 0 1 0 0 0
1 0 1 0 0 0
1 0 1 0 0 0

top 2 * label_mask =
0 1 0 0 0 0
1 0 0 0 0 0
0 0 0 1 0 0
0 0 1 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0

top 1 =
0 1 0 0 0 0
1 0 0 0 0 0
0 0 0 1 0 0
0 0 1 0 0 0 
0 0 1 0 0 0
0 0 1 0 0 0

top 1 * label_mask = 
0 1 0 0 0 0
1 0 0 0 0 0
0 0 0 1 0 0
0 0 1 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
'''