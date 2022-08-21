"""
 Copyright (c) 2022 Intel Corporation
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

from enum import Enum

from nncf.api.compression import ModelType


class BackendType(Enum):
    """
    Contains the list of supported backends.
    """

    OPENVINO = 'OpenVINO'


def determine_backend(model: ModelType) -> BackendType:
    """
    Determines the NNCF backend using the type of passed model.

    :param model: The framework-specific model to be applied
        compression algorithms.
    :return: A backend type that represents the framework.
    """

    try:
        import openvino
    except ImportError:
        openvino = None

    if openvino is not None and isinstance(model, openvino.runtime.Model):
        return BackendType.OPENVINO

    raise RuntimeError(
        'Could not determine the backend framework from the model type '
        'because the framework is not available or the model type is unsupported.'
    )
