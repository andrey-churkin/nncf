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

import sys
from typing import Optional, Callable, Any, Iterable

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.data import Dataset
from nncf.common.quantization.structs import QuantizationPreset
from nncf.scopes import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm  import PostTrainingQuantizationParameters
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_OV_CATEGORY
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer


@tracked_function(NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset"])
def quantize_impl(model: ov.Model,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType] = None,
                  ignored_scope: Optional[IgnoredScope] = None,
                  compress_weights: bool = True) -> ov.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend via the OpenVINO Runtime API.
    """

    quantization_parameters = PostTrainingQuantizationParameters(
        preset=preset,
        target_device=target_device,
        number_samples=subset_size,
        ignored_scopes=ignored_scope,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type
    )

    quantization_algorithm = PostTrainingQuantization(quantization_parameters)
    quantized_model = quantization_algorithm.apply(model, dataset=calibration_dataset)
    if compress_weights:
        compress_quantize_weights_transformation(quantized_model)

    return quantized_model


def quantize_with_accuracy_control(model: ov.Model,
                                   calibration_dataset: Dataset,
                                   validation_dataset: Dataset,
                                   validation_fn: Callable[[Any, Iterable[Any]], float],
                                   max_drop: float = 0.01,
                                   preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                                   target_device: TargetDevice = TargetDevice.ANY,
                                   subset_size: int = 300,
                                   fast_bias_correction: bool = True,
                                   model_type: Optional[ModelType] = None,
                                   ignored_scope: Optional[IgnoredScope] = None,
                                   max_num_iterations: int = sys.maxsize) -> ov.Model:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend via the
    OpenVINO Runtime API.
    """
    quantized_model = quantize_impl(model, calibration_dataset, preset, target_device, subset_size,
                                    fast_bias_correction, model_type, ignored_scope, compress_weights=False)
    accuracy_restorer = QuantizationAccuracyRestorer(max_num_iterations=max_num_iterations,
                                                     max_drop=max_drop)
    quantized_model = accuracy_restorer.apply(model, quantized_model, validation_dataset, validation_fn)
    compress_quantize_weights_transformation(quantized_model)

    return quantized_model
