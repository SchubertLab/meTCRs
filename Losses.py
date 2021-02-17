import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from Utils import UtilsComputation as Compute


def bce_loss(y_actual, y_pred):
    distance_matrix = Compute.pairwise_distance(y_pred)
    mask = 1 - Compute.label_mask(y_actual)
    loss = tf.keras.losses.BinaryCrossentropy()(mask, distance_matrix)
    return loss


def contrastive_loss_at_margin(margin_neg, margin_pos=0., weight_pos=1.):
    def contrastive_loss_batch(y_actual, y_pred):
        distance_matrix = Compute.pairwise_distance(y_pred)
        mask = Compute.label_mask(y_actual)
        loss_pos, loss_neg = contrastive_loss_matrix(distance_matrix, mask, margin_neg, margin_pos)
        loss = loss_neg + weight_pos * loss_pos
        return loss
    return contrastive_loss_batch


def triplet_loss(y_actual, y_pred):
    distances = Compute.pairwise_distance(y_pred)
    anchor_pos = tf.expand_dims(distances, 2)
    anchor_neg = tf.expand_dims(distances, 1)
    loss = anchor_pos - anchor_neg + 0.5
    mask = Compute.triplet_mask(y_actual)
    loss = tf.multiply(loss, mask)
    loss = tf.math.maximum(loss, 0.)

    valid_triplets = tf.cast(tf.greater(loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    loss = tf.reduce_sum(loss) / (num_positive_triplets + 1e-16)

    return loss


def contrastive_loss_matrix(distance_matrix, label_matrix, margin_neg, margin_pos):
    loss_pos = label_matrix * tf.math.maximum(distance_matrix - margin_pos, 0.)
    loss_neg = (1-label_matrix) * tf.math.maximum(margin_neg - distance_matrix, 0.)
    return loss_pos, loss_neg


def semi_hard_triplet_loss():
    return tfa.losses.TripletSemiHardLoss()


@tf.function
def contrastive_loss_tfa(y_actual, y_pred):
    distance_matrix = Compute.pairwise_distance(y_pred)
    mask = Compute.label_mask(y_actual)
    distances = tf.reshape(distance_matrix, shape=[-1])
    mask = tf.reshape(mask, shape=[-1])
    return tfa.losses.contrastive_loss(mask, distances)
