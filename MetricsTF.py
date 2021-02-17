import tensorflow as tf

from Utils import UtilsComputation as Compute


def recall_at_k(k):
    def recall_function(labels, y_pred):
        label_mask = Compute.label_mask(labels)
        label_mask = tf.cast(label_mask, dtype=tf.float32)
        distance_matrix = Compute.pairwise_distance(y_pred)

        # Mask the diagonal with high value s.t. self distance is not chosen
        # -1 * => minimal values become maximal values
        distances_masked = -1. * (distance_matrix + tf.eye(tf.shape(labels)[0]) * 1e10)

        _, indices_top_k = tf.math.top_k(distances_masked, k=k)

        # Create mask of shape label mask with ones at top k indices per row
        mask_top_k = tf.one_hot(indices_top_k, depth=tf.shape(labels)[0])
        mask_top_k = tf.reduce_sum(mask_top_k, axis=-2)

        # combine label and prediction mask and get percentage of if correct prediction in top k
        correct_predictions_at_k = label_mask * mask_top_k
        correct_predictions_at_k = tf.math.reduce_sum(correct_predictions_at_k, axis=-1)
        recall_value = tf.math.count_nonzero(correct_predictions_at_k, dtype=tf.float32)

        recall_value /= tf.cast(tf.shape(labels)[0], dtype=tf.float32)
        # raise ValueError('Breakpoint')
        return recall_value
    recall_function.__name__ = f'R@{k}'
    return recall_function


def contrastive_positive():
    def fn(y_actual, y_pred):
        distance_matrix = Compute.pairwise_distance(y_pred)
        mask = Compute.label_mask(y_actual)
        return distance_matrix * mask
    fn.__name__ = 'L_pos'
    return fn


def contrastive_negative():
    def fnc(y_actual, y_pred):
        distance_matrix = Compute.pairwise_distance(y_pred)
        mask = Compute.label_mask(y_actual)
        loss_neg = (1-mask) * tf.math.maximum(1-distance_matrix, 0.)
        return loss_neg
    fnc.__name__ = 'L_neg'
    return fnc
