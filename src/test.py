from supcon import contrastive_loss, ContrastiveLoss
import tensorflow as tf
import numpy as np

y_true = tf.one_hot([0, 1, 0, 1, 0], 2)
print(y_true)
loss = ContrastiveLoss()(y_true, np.random.normal(0, 1, (5, 10)))
print(loss)