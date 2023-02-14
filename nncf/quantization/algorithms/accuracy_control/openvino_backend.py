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

from typing import List
from typing import Any

import openvino.runtime as ov

from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.experimental.openvino_native.graph.metatypes.common import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.common import QUANTIZABLE_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.common import FAKE_QUANTIZE_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.common import CONSTANT_OPERATIONS
from nncf.experimental.openvino_native.graph.transformations.command_creation import create_command_to_remove_quantizer
from nncf.experimental.openvino_native.graph.transformations.command_creation import create_bias_correction_command
from nncf.experimental.openvino_native.graph.node_utils import is_node_with_bias
from nncf.experimental.openvino_native.graph.node_utils import get_bias_value


class OVAccuracyControlAlgoBackend(AccuracyControlAlgoBackend):
    """
    Implementation of the `AccuracyControlAlgoBackend` for OpenVINO backend.
    """

    # Metatypes

    @staticmethod
    def get_quantizer_metatypes() -> List[OVOpMetatype]:
        return FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def get_const_metatypes() -> List[OVOpMetatype]:
        return CONSTANT_OPERATIONS

    @staticmethod
    def get_quantizable_metatypes() -> List[OVOpMetatype]:
        return QUANTIZABLE_OPERATIONS

    @staticmethod
    def get_quantize_agnostic_metatypes() -> List[OVOpMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS

    # Creation of commands

    @staticmethod
    def create_command_to_remove_quantizer(quantizer_node: NNCFNode):
        return create_command_to_remove_quantizer(quantizer_node)

    @staticmethod
    def create_command_to_update_bias(node_with_bias: NNCFNode, bias_value: Any, nncf_graph: NNCFGraph):
        return create_bias_correction_command(node_with_bias, bias_value, nncf_graph)

    # Manipulations with bias value

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model) -> Any:
        return get_bias_value(node_with_bias, nncf_graph, model)

    # Preparation of model

    @staticmethod
    def prepare_for_inference(model) -> Any:
        return ov.compile_model(model)