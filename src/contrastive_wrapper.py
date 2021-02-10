# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""The contrastive model."""

import tensorflow.compat.v1 as tf
from contrastive_heads import ProjectionHead, ClassificationHead


class ContrastiveModel(tf.layers.Layer):
    """A model suitable for contrastive training with different backbone networks.
    Paremeters
    ----------
    transformer: tensorflow network to train
    normalize_projection_head_input: Whether the transformer output that is the
        input to the projection head should be normalized.
    normalize_classification_head_input: Whether the transformer output that is
        the input to the classification head should be normalized.
    stop_gradient_before_projection_head: Whether the projection head is
        trained simultaneously with the transformer. If true, stop_gradient
        is added between the projection head and the transformer.
    stop_gradient_before_classification_head: Whether the classification head
        is trained simultaneously with the transformer. If true, stop_gradient
        is added between the classification head and the transformer.
    projection_head_kwargs: Keyword arguments that are passed on to the
        constructor of the projection head. These are the arguments to
        `projection_head.ProjectionHead`.
    classification_head_kwargs: Keyword arguments that are passed on to the
        constructor of the classification head. These are the arguments to
        `classification_head.ClassificationHead`.
    name: A name for this object.
    """

    def __init__(self,
                 transformer,
                 normalize_projection_head_input=True,
                 normalize_classification_head_input=True,
                 stop_gradient_before_projection_head=False,
                 stop_gradient_before_classification_head=True,
                 transformer_kwargs=None,
                 projection_head_kwargs=None,
                 classification_head_kwargs=None,
                 name='ContrastiveModel',
                 **kwargs):
        super(ContrastiveModel, self).__init__(name=name, **kwargs)

        self.normalize_projection_head_input = normalize_projection_head_input
        self.normalize_classification_head_input = (
            normalize_classification_head_input)
        self.stop_gradient_before_projection_head = (
            stop_gradient_before_projection_head)
        self.stop_gradient_before_classification_head = (
            stop_gradient_before_classification_head)

        projection_head_kwargs = projection_head_kwargs or {}
        classification_head_kwargs = classification_head_kwargs or {}

        self.transformer = transformer
        self.projection_head = ProjectionHead(
            **projection_head_kwargs)
        self.classification_head = ClassificationHead(
            **classification_head_kwargs)

    def call(self, inputs, training):
        embedding = self.transformer(inputs, training)
        normalized_embedding = tf.nn.l2_normalize(embedding, axis=1)

        projection_input = (
            normalized_embedding
            if self.normalize_projection_head_input else embedding)
        if self.stop_gradient_before_projection_head:
            projection_input = tf.stop_gradient(projection_input)
        projection = self.projection_head(projection_input, training)

        classification_input = (
            normalized_embedding
            if self.normalize_classification_head_input else embedding)
        if self.stop_gradient_before_classification_head:
            classification_input = tf.stop_gradient(classification_input)
        classification = self.classification_head(
            classification_input, training)

        return embedding, normalized_embedding, projection, classification
