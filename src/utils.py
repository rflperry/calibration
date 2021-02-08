import tensorflow as tf


def tanh_hard(x):
    """ Hard tanh """
    return tf.minimum(1.0, tf.maximum(0.0, x))


def predict_proba(logits, temp=1):
    probs = tf.nn.softmax(logits / temp).numpy()
    return probs.argmax(1)
