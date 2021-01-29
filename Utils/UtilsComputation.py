import tensorflow as tf


def label_mask(labels):
    labels = tf.expand_dims(labels, -1)
    mask = tf.math.equal(labels, tf.transpose(labels))
    mask = tf.cast(mask, dtype=tf.dtypes.float32)
    return mask


def pairwise_distance(vector):
    # D[i,j] = (a[i]-a[j])(a[i]-a[j])'
    # D[i,j] = r[i] - 2 a[i]a[j]' + r[j]
    r = tf.reduce_sum(vector*vector, 1)
    r = tf.reshape(r, [-1, 1])
    distance_matrix = r - 2 * tf.matmul(vector, tf.transpose(vector)) + tf.transpose(r)
    distance_matrix = tf.maximum(distance_matrix, 0.)
    error_mask = tf.math.less_equal(distance_matrix, 0.)
    distance_matrix = tf.math.sqrt(distance_matrix + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)
    return distance_matrix
