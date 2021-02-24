import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class Linear(layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="glorot_uniform", trainable=True
        )

    def call(self, inputs):
        out = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            out = self.activation(out)
        return out

    def margin_distances(self, x, positive=True):
        dists = tf.matmul(x, self.w) + self.b
        return dists


class ReluNet(layers.Layer):
    """Simple stack of Linear layers."""

    def __init__(self, layer_sizes, act=tf.keras.activations.relu,
                 normed_output=False, out_act=None):
        super().__init__()
        self.layers = [Linear(units, act) for units in layer_sizes[:-1]]
        if out_act:
            self.layers.append(Linear(layer_sizes[-1], out_act))
        else:
            self.layers.append(Linear(layer_sizes[-1], act))
        self.act = act
        self.out_act = out_act
        self.normed_output = normed_output

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.normed_output:
            x, _ = tf.linalg.normalize(x, axis=1)
        return x

    def activations(self, x):
        acts = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            bool_mat = x.numpy() > 0
            acts.append(bool_mat.astype(int))
        return acts

    def margin_distances(self, x, positive=True):
        dists = []
        for i, layer in enumerate(self.layers):
            x = layer.margin_distances(x)
            # since matmult is out = x @ w + b
            dist = x / tf.norm(layer.w, axis=0, keepdims=True)
            if positive:
                dist = tf.math.abs(dist)
            dists.append(dist)
        dists = tf.concat(dists, axis=1)
        dists = tf.squeeze(dists)
        # dists = tf.reduce_min(dists, axis=1)
        return dists


class EmbeddingNet(layers.Layer):
    """
    Embeds the data in some output space. Can then output the embeddings
    or classify by adding a classification layer
    """

    def __init__(
        self,
        model=None,
        embed_dim=2,
        n_classes=None,
        activation=tf.nn.relu,
        norm_embedding=False,
        projection_head=True,
    ):
        super().__init__()
        if model is None:
            self.embeddor = create_LeNet(out_dim=embed_dim)
        else:
            self.embeddor = model
        self.activation = activation
        self.n_classes = n_classes
        self.norm_embedding = norm_embedding
        if n_classes is not None:
            self.classifier = Linear(n_classes)
        if projection_head:
            self.projection_head = create_projection_head(128)
        else:
            self.projection_head = None

    def call(self, inputs):
        output = self.get_embedding(inputs)
        if self.n_classes is not None:
            output = self.activation(output)
            output = self.classifier(output)
        elif self.projection_head is not None:
            output = self.get_projection(output)

        return output

    def get_embedding(self, x):
        embedding = self.embeddor(x)
        if self.norm_embedding:
            embedding, _ = tf.linalg.normalize(embedding, axis=1)
        return embedding

    def get_projection(self, x):
        projection = self.projection_head(x)
        if self.norm_embedding:
            embedding, _ = tf.linalg.normalize(projection, axis=1)
        return projection


def create_projection_head(out_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(out_dim))

    return model


def create_LeNet(out_dim=2):
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(out_dim))

    return model
