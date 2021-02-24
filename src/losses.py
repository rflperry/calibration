import tensorflow as tf
import numpy as np


def margin_loss(margins, activations, y):
    region = np.unique(np.hstack(activations), axis=0, return_inverse=True)[1]
    region_losses = []
    y = y.numpy()
    for r in np.unique(region):
        idx = np.where(region == r)[0]
        y_pred = np.bincount(y[idx]).argmax()
        misclassified_idx = idx[np.where(y_pred != y[idx])[0]]

        # collect margins of wrong
        margins_in_region = tf.gather(margins, misclassified_idx, axis=0)
        min_margins = tf.reduce_min(margins_in_region, axis=1)
        region_losses.append(tf.reduce_sum(min_margins))

        # region_losses.append(tf.reduce_sum(margins))

    # sum over losses in all regions, from missclassified samples
    return tf.reduce_sum(region_losses)


# Like margin loss. Distance penalized up to a certain point
# Given a metric (l2) and two samples, train to produce similar representations
# if inputs are similar (i.e. same class), or dissimilar representations if
# inputs are dissimilar.
#
# Given representations r_0, r_1, flag y = 1 if similar or 0 if dissimilar, and
# maxmimum margin penalty m, the loss is
# L(r_0, r_1, y) = y||r_0 - r_1||_2^2 + (1-y)max(0, m - ||r_0 - r_1||_2^2)
def pairwise_ranking_loss(r0, r1, y, margin=1):
    dists = tf.norm(r0 - r1, ord='euclidean', axis=1)
    loss = y * dists + (1 - y) * tf.maximum(0.0, margin - dists)
    loss = tf.square(loss)
    return tf.reduce_mean(loss)


# Like pairwise_ranking_loss, but all pairwise distances
# requires one hot encoded y
def batch_contrastive_loss(r, y, margin=1):
    r_norms = tf.reduce_sum(r * r, 1)
    # turn r_norms into column vector
    r_norms = tf.reshape(r_norms, [-1, 1])
    r_dists = r_norms - 2*tf.matmul(r, tf.transpose(r)) + tf.transpose(r_norms)
    r_dists = tf.nn.relu(r_dists)  # avoid numerical issues
    r_dists = tf.linalg.set_diag(
        r_dists, tf.zeros(r_dists.shape[0], dtype=r_dists.dtype)
        )  # exclude itself distance
    r_dists = tf.sqrt(r_dists)
    r_dists = tf.reshape(r_dists, [-1])

    n_classes = y.shape[1]
    y_norms = tf.reduce_sum(y * y, 1)

    y = y_norms - 2*tf.matmul(y, tf.transpose(y)) + tf.transpose(y_norms)
    y = 1 - y
    y = tf.reshape(y, [-1])
    y = tf.cast(y / n_classes, r_dists.dtype)

    loss = y * r_dists + (1 - y) * tf.maximum(0, margin - r_dists)
    loss = tf.square(loss)
    return tf.reduce_mean(loss)


# sum over samples in batch (anchors) ->
# average over similar samples (positive) ->
# of - log softmax positive / sum negatives (wrt cos similarity)
# i.e. \sum_i -1/|P(i)| \sum_{p \in P(i)} log [exp(z_i @ z_p / t) / \sum_{n \in N(i)} exp(z_i @ z_n / t)]
# = \sum_i [log[\sum_{n \in N(i)} exp(z_i @ z_n / t)] - 1/|P(i)| \sum_{p \in P(i)} log [exp(z_i @ z_p / t)]]
def supervised_contrastive_loss(yTrue, yPred, temp=0.1):
    r = yPred
    y = yTrue
    r, _ = tf.linalg.normalize(r, axis=1)

    r_dists = tf.matmul(r, tf.transpose(r))
    r_dists = tf.linalg.set_diag(
        r_dists, tf.zeros(r_dists.shape[0], dtype=r_dists.dtype)
    )  # exclude itself distance
    r_dists = r_dists / temp

    y_norms = tf.reduce_sum(y * y, 1)
    y = y_norms - 2*tf.matmul(y, tf.transpose(y)) + tf.transpose(y_norms)

    y = tf.cast(y / 2, r_dists.dtype)  # scale onehot distances to 0 and 1
    negative_sum = tf.math.log(
        tf.reduce_sum(y * tf.exp(r_dists), axis=1))  # y zeros diagonal 1's
    positive_sum = (1 - y) * r_dists

    n_nonzero = tf.math.reduce_sum(1-y, axis=1) - 1  # Subtract diagonal
    positive_sum = tf.reduce_sum(
        positive_sum, axis=1
        ) / tf.cast(n_nonzero, positive_sum.dtype)
    loss = tf.reduce_sum(negative_sum - positive_sum)

    return loss


# ~ p(x_i in leaf) * p(y_hat == y_i)
def pseudo_entropy_loss(margins, activations, y):
    region = np.unique(np.hstack(activations), axis=0, return_inverse=True)[1]
    region_losses = []
    for r in np.unique(region):
        idx = np.where(region == r)[0]
        y_probs = np.bincount(y[idx]) / len(y[idx])
        y_pred = np.bincount(y[idx]).argmax()
        misclass_idx = idx[np.where(y_pred != y[idx])[0]]
        class_idx = idx[np.where(y_pred == y[idx])[0]]

        misclass_margins = tf.gather(margins, misclass_idx, axis=0)
        min_misclass_margins = tf.reduce_min(misclass_margins, axis=1)
        misclass_probs = [y_probs[cls] for cls in y[misclass_idx]]
        weighted_misclass_margins = tf.multiply(
            min_misclass_margins, misclass_probs)

        class_margins = tf.gather(margins, class_idx, axis=0)
        min_class_margins = tf.reduce_min(class_margins, axis=1)
        class_probs = [y_probs[cls] for cls in y[class_idx]]
        weighted_class_margins = tf.multiply(
            min_class_margins, class_probs)

        region_losses.append(
            tf.reduce_sum(weighted_misclass_margins) -
            tf.reduce_sum(weighted_class_margins)
        )

    # sum over losses in all regions
    return tf.reduce_sum(region_losses)


def softmax_loss(logits, y, temp=1):
    logits = logits / temp
    if isinstance(y, (np.ndarray, list)):
        y = tf.one_hot(y, depth=len(np.unique(y)))
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(losses)
