import tensorflow as tf
import numpy as np


def spiral(n=100):
    N = n
    from numpy import pi
    theta = np.sqrt(np.random.rand(N)) * 2 * pi  # np.linspace(0,2*pi,100)

    r_a = 2*theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N, 2)

    r_b = -2*theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(N, 2)

    res_a = np.append(x_a, np.zeros((N, 1)), axis=1)
    res_b = np.append(x_b, np.ones((N, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    return np.vstack((x_a, x_b)), np.hstack(([0]*len(x_a), [1]*len(x_b)))


class PairwiseLoader():
    """
    Train: For each sample creates randomly a positive or a negative pair
    TODO Test: Creates fixed pairs for testing

    Adopted from:
    https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.train_labels = y_train
        self.train_data = X_train
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        X1, y1 = self.train_data[index], self.train_labels[index]
        if target == 1:
            paired_index = index
            while paired_index == index:
                paired_index = np.random.choice(self.label_to_indices[y1])
        else:
            paired_label = np.random.choice(
                list(self.labels_set - set([y1])))
            paired_index = np.random.choice(
                self.label_to_indices[paired_label])
        X2 = self.train_data[paired_index]

#         if self.transform is not None:
#             X1 = self.transform(X1)
#             X2 = self.transform(X2)
        return X1, X2, target

    def __len__(self):
        return len(self.train_data)
