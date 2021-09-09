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

import inspect
from operator import attrgetter
from typing import Dict, Any, List

import tensorflow as tf

from nncf.tensorflow.quantization.algorithm import TFQuantizerSpec

from nncf.tensorflow.experimental.nncf_network import NNCFNetwork
from nncf.tensorflow.experimental.wrappers import TFHook
from nncf.tensorflow.experimental.wrappers import wrap_tf_operation
from nncf.tensorflow.experimental.quantization.ops import SymmetricQuantizer


class TFGraphPoint:
    """
    Describes a place in the `tf.Graph` where we should place
    the NNCF operation.
    """

    def __init__(self,
                 op_name: str,
                 op_type_name: str,
                 port_id: int,
                 is_pre_hook: bool):
        """
        Initializes the `TFGraphPoint`.

        :param op_name: Name of operation in `tf.Graph`.
        :param op_type_name: Name of operation type.
        :param port_id: Zero-based index
        :param is_pre_hook: `True`/`False` if we should place NNCF operation
            before/after TensorFlow operation.
        """
        self.op_name = op_name
        self.op_type_name = op_type_name
        self.port_id = port_id
        self.is_pre_hook = is_pre_hook

    def get_state(self) -> Dict[str, Any]:
        state = {
            'op_name': self.op_name,
            'op_type_name': self.op_type_name,
            'port_id': self.port_id,
            'is_pre_hook': self.is_pre_hook,
        }
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'TFGraphPoint':
        return cls(**state)


class QuantizerDesc:
    def __init__(self,
                 name: str,
                 quantizer_spec: TFQuantizerSpec,
                 target_ops: List[TFGraphPoint],
                 input_shape: List[int],
                 scale_val: float = 6.0,
                 sign_val: float = -1.0,
                 priority: int = 0):
        """
        Initializes description of quantizer.
        """
        self.name = name
        self.quantizer_spec = quantizer_spec
        self.target_ops = target_ops
        self.input_shape = input_shape
        self.scale_val = scale_val
        self.sign_val = sign_val
        self.priority = priority

    def get_state(self) -> Dict[str, Any]:
        state = {
            'name': self.name,
            'quantizer_spec': self.quantizer_spec.get_state(),
            'target_ops': [op.get_state() for op in self.target_ops],
            'input_shape': self.input_shape,
            'scale_val': self.scale_val,
            'sign_val': self.sign_val,
            'priority': self.priority,
        }
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'QuantizerDesc':
        quantizer_spec = TFQuantizerSpec.from_state(state['quantizer_spec'])

        target_ops = [
            TFGraphPoint.from_state(item) for item in state['target_ops']
        ]
        return cls(state['name'], quantizer_spec, target_ops, state['input_shape'],
                   state['scale_val'], state['sign_val'], state['priority'])


def _get_unique_op_type_names(descriptions: List[QuantizerDesc]) -> List[str]:
    op_type_names = set()
    for q_desc in descriptions:
        for op_desc in q_desc.target_ops:
            op_type_names.add(op_desc.op_type_name)
    return list(op_type_names)


def _get_ops_info(op_type_names: List[str]):
    raw_ops = inspect.getmembers(tf.raw_ops, predicate=inspect.isfunction)
    op_type_name_to_fn_map = dict(raw_ops)

    ops_info = {}
    for op_type_name in op_type_names:
        original_fn = op_type_name_to_fn_map[op_type_name]

        module = inspect.getmodule(original_fn)
        fn_name = original_fn.__name__
        fn = getattr(module, fn_name)

        ops_info[op_type_name] = (fn, fn_name, module)

    return ops_info


def quantize_model(model: tf.keras.Model, descriptions: List[QuantizerDesc]) -> NNCFNetwork:
    nncf_network = NNCFNetwork(model)

    _names = []

    hooks = {}
    for qdesc in descriptions:

        assert qdesc.quantizer_spec.mode == 'symmetric'
        assert qdesc.name not in _names
        _names.append(qdesc.name)

        quant = SymmetricQuantizer(
            qdesc.name,
            qdesc.quantizer_spec,
            qdesc.scale_val,
            qdesc.sign_val
        )

        nncf_network.add_operation(quant)

        assert len(qdesc.target_ops) == 1

        for op in qdesc.target_ops:
            op_hooks = hooks.setdefault(op.op_name, [])
            op_hooks.append(
                TFHook(quant, op.port_id, op.is_pre_hook, op.op_type_name, qdesc.priority)
            )

    pre_hooks = {}
    post_hooks = {}
    for op_name, elems in hooks.items():
        pre_hooks[op_name] = sorted(
            (hook for hook in elems if hook.is_pre_hook),
            key=attrgetter('priority'),
            reverse=True
        )
        post_hooks[op_name] = sorted(
            (hook for hook in elems if not hook.is_pre_hook),
            key=attrgetter('priority'),
            reverse=True
        )

    unique_op_type_names = _get_unique_op_type_names(descriptions)
    ops_info = _get_ops_info(unique_op_type_names)

    for op_type_name in unique_op_type_names:
        fn, fn_name, module = ops_info[op_type_name]
        setattr(module, fn_name, wrap_tf_operation(fn, op_type_name, pre_hooks, post_hooks))

    return nncf_network
