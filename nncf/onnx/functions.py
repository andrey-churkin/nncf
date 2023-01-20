"""
 Copyright (c) 2023 Intel Corporation
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

import onnx
import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.graph import NNCFNode
from nncf.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXIdentityMetatype
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand


def is_node_with_bias(node: NNCFNode) -> bool:
    """
    :param node:
    :return:
    """
    input_tensor_names = node.layer_attributes.input_tensor_names
    return node.metatype in LAYERS_WITH_BIAS_METATYPES and len(input_tensor_names) > 2


def get_bias_value(node: NNCFNode, model: onnx.ModelProto) -> np.ndarray:
    """
    :param node:
    :param model:
    :return:
    """
    onnx_graph = ONNXGraph(model)
    onnx_node = onnx_graph.get_node_by_name(node.node_name)
    bias_port_id = onnx_graph.get_bias_tensor_port_id(onnx_node)
    bias_input_name = onnx_node.input[bias_port_id]
    if onnx_graph.has_initializer(bias_input_name):
        return onnx_graph.get_initializers_value(bias_input_name)
    node = onnx_graph.get_nodes_by_output(bias_input_name)[0]
    metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
    if metatype == ONNXIdentityMetatype:
        return onnx_graph.get_initializers_value(node.input[0])
    raise RuntimeError('Could not find the bias value of the node')


def create_bias_correction_command(node: NNCFNode, bias_value: np.ndarray) -> ONNXBiasCorrectionCommand:
    """
    :param node:
    :param bias_value:
    :return:
    """
    bias_port_id = node.metatype.weight_definitions.bias_port_id
    target_point = ONNXTargetPoint(TargetType.LAYER, node.node_name, bias_port_id)
    return ONNXBiasCorrectionCommand(target_point, bias_value)
