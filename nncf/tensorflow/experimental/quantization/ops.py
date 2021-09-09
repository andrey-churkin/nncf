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

from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.functions import symmetric_quantize


# It's the simplified version of the `nncf.tensorflow.quantization.quantizers.SymmetricQuantizer` class.
class SymmetricQuantizer(Quantizer):
    def __init__(self,
                 name: str,
                 qspec: TFQuantizerSpec,
                 scale_val: float,
                 sign_val: float):
        super().__init__(name)
        self.num_bits = qspec.num_bits
        self.per_channel = qspec.per_channel
        self.narrow_range = qspec.narrow_range
        self.signedness_to_force = qspec.signedness_to_force
        self.half_range = qspec.half_range
        self.scale_val = scale_val
        self.sign_val = sign_val

        if self.per_channel:
            raise RuntimeError('Only per-tensor mode is supported.')

    def build(self, layer):
        with tf.name_scope(self.name):
            self._scale_var = layer.add_weight(
                'scale',
                shape=None,
                initializer=tf.keras.initializers.Constant(self.scale_val),
                trainable=True
            )
            self._sign_var = layer.add_weight(
                'sign',
                initializer=tf.keras.initializers.Constant(self.sign_val),
                trainable=False
            )

    def quantize(self, inputs, weights=None, training=None):
        num_bits = self.num_bits - 1 if self.half_range else self.num_bits
        return symmetric_quantize(
                inputs,
                self._scale_var,
                self._sign_var,
                num_bits=num_bits,
                per_channel=self.per_channel,
                narrow_range=self.narrow_range,
                eps=self._eps,
                name_prefix=f'{self.name}/SymmQuant'
            )
