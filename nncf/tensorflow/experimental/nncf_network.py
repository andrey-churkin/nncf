"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf

from nncf.tensorflow.experimental.nncf_context import get_nncf_context


class NNCFNetwork(tf.keras.Model):
    """
    Wraps the tf.keras.Model.
    """

    def __init__(self, model: tf.keras.Model, **kwargs):
        """
        Initializes the NNCF network.

        :param model: A tf.keras.Model.
        """
        super().__init__(**kwargs)
        self._model = model
        self.__dict__['_nncf_ops'] = []

    @property
    def nncf_ops(self):
        return getattr(self, '_nncf_ops')

    def call(self, inputs, **kwargs):
        with get_nncf_context().enter(wrap_ops=True):
            outputs = self._model(inputs, **kwargs)
        return outputs

    def add_operation(self, nncf_op):
        self.nncf_ops.append(nncf_op)

    def build(self, input_shape=None):
        for op in self.nncf_ops:
            op.build(self)
        self._model.build(input_shape)
