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

import json
import argparse

import tensorflow as tf

from nncf.tensorflow.experimental.quantization.algorithm import QuantizerDesc
from nncf.tensorflow.experimental.quantization.algorithm import quantize_model
from nncf.tensorflow.experimental.export import to_frozen_graph


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='name of the model from the tf.keras.applications',
        metavar='MODEL_NAME'
    )

    parser.add_argument(
        '--desc',
        type=str,
        required=True,
        help='path to JSON file with quantizers descriptions',
        metavar='JSON_FILE'
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print('Arguments:')
    print(f'\tmodel_name: {args.model_name}')
    print(f'\tdescs: {args.desc}')

    with open(args.desc) as f:
        states = json.load(f)
    descriptions = [
        QuantizerDesc.from_state(s) for s in states
    ]

    model_fn = getattr(tf.keras.applications, args.model_name)
    model = model_fn()

    nncf_network = quantize_model(model, descriptions)

    input_signature = tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32, name='input_1')
    concrete_function = tf.function(nncf_network).get_concrete_function(input_signature)

    graph = to_frozen_graph(nncf_network, concrete_function)
    tf.io.write_graph(graph, 'tmp_folder', 'model.pb', as_text=False)

if __name__ == '__main__':
    main()
