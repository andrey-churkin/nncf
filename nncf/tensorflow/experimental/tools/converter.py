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
from typing import List

import tensorflow as tf

from nncf.tensorflow.quantization.algorithm import TFQuantizationSetup
from nncf.common.graph.transformations.commands import TargetType

from nncf.tensorflow.experimental.quantization.algorithm import QuantizerDesc
from nncf.tensorflow.experimental.quantization.algorithm import TFGraphPoint


CFG = {
    'Relu6': {
        'output': {
            'tensor': 0,
        }
    },
    'FusedBatchNormV3': {
        'output': {
            'tensor': 0,
        }
    },
    'AddV2': {
        'output': {
            'tensor': 0,
        }
    },
    'Mean': {
        'output': {
            'tensor': 0,
        }
    },
    'Conv2D': {
        'input': {
            'tensor': 0,
            'weight': 1,
        },
        'output': {
            'tensor': 0,
        }
    },
    'DepthwiseConv2dNative': {
        'input': {
            'tensor': 0,
            'weight': 1,
        },
        'output': {
            'tensor': 0,
        }
    },
    'MatMul': {
        'input': {
            'tensor': 0,
            'weight': 1,
        },
        'output': {
            'tensor': 0,
        }
    }
}


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
        '--qs',
        type=str,
        required=True,
        help='path to JSON file with quantizer setup',
        metavar='JSON_FILE'
    )

    parser.add_argument(
        '--wq',
        type=str,
        required=True,
        help='parameters for weight quantizers',
        metavar='JSON_FILE'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='path to JSON file where the result should be saved',
        metavar='JSON_FILE'
    )

    return parser


def get_tf_graph(model_name: str):
    model_name_to_model_fn = {
        'MobileNetV2': tf.keras.applications.MobileNetV2,
    }

    model_fn = model_name_to_model_fn[model_name]

    # MobileNetV2
    model = model_fn(input_shape=(224, 224, 3), classes=1000)
    input_signature = tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='input_1')
    concrete_function = tf.function(model).get_concrete_function(input_signature)
    return concrete_function.graph


def get_layer_name(name: str):
    """
    Returns name of a layer from the name of TF operation.

    :param name: Name of operation.
    :return: Name of layer.
    """
    layer_name = name[name.find('/') + 1:]
    pos = layer_name.find('/')
    if pos != -1:
        layer_name = layer_name[:layer_name.find('/')]
    return layer_name


def _get_target_ops_and_input_shape(ops, graph_ops, is_pre_hook, is_wq) -> List[TFGraphPoint]:
    target_ops = []

    key = 'input' if is_pre_hook else 'output'

    if is_wq:
        if len(ops) > 1:
            print(ops)
        op = ops[0]
        port_id = CFG[op.type][key]['weight']
    else:
        assert len(ops) == 1
        op = ops[0]
        port_id = CFG[op.type][key]['tensor']

    if is_pre_hook:
        input_shape = op.inputs[port_id].shape.as_list()
    else:
        input_shape = op.outputs[port_id].shape.as_list()

    target_ops.append(TFGraphPoint(op.name, op.type, port_id, is_pre_hook))

    return input_shape, target_ops


def run_converter(model_name: str, quantization_setup: TFQuantizationSetup) -> List[QuantizerDesc]:
    tf_graph = get_tf_graph(model_name)

    excluded_types = ['ReadVariableOp', 'Placeholder', 'Const']
    tf_graph_ops = [
        op for op in tf_graph.get_operations() if op.type not in excluded_types
    ]

    layer_name_to_ops_map = {}
    for op in tf_graph_ops:
        layer_name = get_layer_name(op.name)
        items = layer_name_to_ops_map.setdefault(layer_name, [])
        items.append(op)

    quantizers = []
    for quantization_point in quantization_setup:
        nncf_op_name = quantization_point.op_name
        quantizer_spec = quantization_point.quantizer_spec

        target_point_type = quantization_point.target_point.type
        if target_point_type == TargetType.AFTER_LAYER:
            is_pre_hook = False
        elif target_point_type == TargetType.OPERATION_WITH_WEIGHTS:
            is_pre_hook = True
        else:
            RuntimeError(f'Unexpected type of target point: {target_point_type}')

        layer_name = quantization_point.target_point.layer_name
        if layer_name == 'input_1':
            continue

        is_wq = quantization_point.is_weight_quantization()
        input_shape, target_ops = _get_target_ops_and_input_shape(layer_name_to_ops_map[layer_name],
                                                                  tf_graph_ops,
                                                                  is_pre_hook,
                                                                  is_wq)
        quantizers.append(
            QuantizerDesc(
                name=nncf_op_name,
                quantizer_spec=quantizer_spec,
                target_ops=target_ops,
                input_shape=input_shape,
                priority=0
            )
        )

    return quantizers


def main():
    parser = create_parser()
    args = parser.parse_args()

    print('Arguments:')
    print(f'\tmodel_name: {args.model_name}')
    print(f'\tquantizer_setup: {args.qs}')
    print(f'\twq params: {args.wq}')
    print(f'\toutput_file: {args.output_file}')

    with open(args.qs) as f:
        state = json.load(f)
    quantization_setup = TFQuantizationSetup.from_state(state)

    descriptions = run_converter(args.model_name, quantization_setup)

    with open(args.wq) as f:
        wq_params = json.load(f)

    for item in wq_params:
        matches_cnt = 0
        for desc in descriptions:
            if item['name'] == desc.name:
                desc.scale_val = item['scale_val']
                desc.sign_val = item['sign_val']
                matches_cnt += 1
        assert matches_cnt == 1

    with open(args.output_file, 'w') as f:
        data = [qd.get_state() for qd in descriptions]
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
