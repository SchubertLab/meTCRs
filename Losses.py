import tensorflow as tf

from Utils import UtilsComputation as Compute


def contrastive_loss_at_margin(margin):
    def contrastive_loss_batch(y_actual, y_pred):
        distance_matrix = Compute.pairwise_distance(y_pred)
        mask = Compute.label_mask(y_actual)
        loss_matrix = contrastive_loss_matrix(distance_matrix, mask, margin)
        loss = tf.reduce_mean(loss_matrix)
        return loss
    return contrastive_loss_batch


def contrastive_loss_matrix(distance_matrix, label_matrix, margin):
    loss_pos = distance_matrix * label_matrix
    loss_neg = (1-label_matrix) * tf.math.maximum(margin-distance_matrix, 0.)
    loss_total = loss_pos + loss_neg
    return loss_total
