import tensorflow as tf


def label_mask(labels):
    labels = tf.expand_dims(labels, -1)
    mask = tf.math.equal(labels, tf.transpose(labels))
    mask = tf.cast(mask, dtype=tf.dtypes.float32)
    return mask


def pairwise_distance(vector):
    # D[i,j] = (a[i]-a[j])(a[i]-a[j])'
    # D[i,j] = r[i] - 2 a[i]a[j]' + r[j]
    r = tf.reduce_sum(vector*vector, -1)
    r = tf.reshape(r, [-1, 1])
    distance_matrix = r - 2 * tf.matmul(vector, tf.transpose(vector)) + tf.transpose(r)
    distance_matrix = tf.maximum(distance_matrix, 0.)
    error_mask = tf.cast(tf.math.equal(distance_matrix, 0.), dtype=tf.dtypes.float32)
    distance_matrix = tf.math.sqrt(distance_matrix + error_mask * 1e-16)
    distance_matrix = distance_matrix * (1-error_mask)

    # zero out self distance
    num_data = tf.shape(vector)[0]
    mask_diag = tf.ones_like(distance_matrix) - tf.linalg.diag(tf.ones([num_data]))
    distance_matrix = tf.multiply(distance_matrix, mask_diag)
    return distance_matrix


def pairwise_l1_distance(vector):
    # todo?
    r = vector - tf.transpose(vector)
    r = tf.abs(r)
    r = tf.reduce_sum(r, axis=-1)
    return r


def triplet_mask(labels):
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)
    return mask
