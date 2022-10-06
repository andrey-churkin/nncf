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

import tempfile
from pathlib import Path
from typing import Optional
from typing import Callable

import openvino
from openvino.tools import pot

from nncf.ptq.data.dataloader import NNCFDataLoader
from nncf.ptq.openvino.dataloader import SizedNNCFDataLoaderImpl
from nncf.ptq.openvino.engine import CustomEngine


def _convert_openvino_model_to_compressed_model(model: openvino.runtime.Model,
                                                target_device: str) -> pot.graph.nx_model.CompressedModel:
    """
    Serializes the provided OpenVINO model and loads the model in the POT representation.

    :param model: The OpenVINO model.
    :param target_device: The target device.
    :return: The POT representation of the provided model.
    """
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        xml_path = str(Path(tmp_dir).joinpath('model.xml'))
        bin_path = str(Path(tmp_dir).joinpath('model.bin'))
        openvino.runtime.serialize(model, xml_path, bin_path)
        model_config = {
            'model_name': 'model',
            'model': xml_path,
            'weights': bin_path,
        }
        pot_model = pot.load_model(model_config, target_device)

    return pot_model


def quantize_impl(model: openvino.runtime.Model,
                  calibration_dataset: NNCFDataLoader,
                  preset: str,
                  target_device: str,
                  subset_size: int,
                  fast_error_correction: bool,
                  model_type: Optional[str] = None) -> openvino.runtime.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend.
    """
    pot_model = _convert_openvino_model_to_compressed_model(model, target_device)

    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 1,
        'eval_requests_number': 1,
    }

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': target_device,
                'preset': preset,
                'stat_subset_size': subset_size,
                'use_fast_bias': fast_error_correction,
                'model_type': model_type,
            }
        }
    ]

    pot_dataloader = SizedNNCFDataLoaderImpl(calibration_dataset,
                                             calibration_dataset._transform_fn,
                                             calibration_dataset.batch_size)
    engine = CustomEngine(engine_config, pot_dataloader, pot_dataloader)
    pipeline = pot.create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(pot_model)
    pot.compress_model_weights(compressed_model)

    compressed_model_paths = pot.save_model(compressed_model, save_path='/tmp/pot', model_name='model')
    ir_model_xml = compressed_model_paths[0]['model']
    ir_model_bin = compressed_model_paths[0]['weights']
    ie = openvino.runtime.Core()
    quantized_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    return quantized_model


def quantize_with_accuracy_control_impl(model: openvino.runtime.Model,
                                        calibration_dataset: NNCFDataLoader,
                                        validation_dataset: NNCFDataLoader,
                                        validation_fn: Callable[[openvino.runtime.Model, NNCFDataLoader], float],
                                        max_drop: float = 0.01,
                                        higher_better: bool = True,
                                        preset: str = 'performance',
                                        target_device: str = 'ANY',
                                        subset_size: int = 300,
                                        fast_error_correction: bool = True,
                                        model_type: Optional[str] = None) -> openvino.runtime.Model:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend.
    """
    pot_model = _convert_openvino_model_to_compressed_model(model, target_device)

    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 1,
        'eval_requests_number': 1,
    }

    algorithms = [
        {
            'name': 'AccuracyAwareQuantization',
            'params': {
                'target_device': target_device,
                'stat_subset_size': subset_size,
                'maximal_drop': max_drop,
                'force_logit_comparison': True,
                'logit_distance_type': 'mse',
                'metric_subset_ratio': 0.5,
                'preset': preset,
                'use_fast_bias': fast_error_correction,
                'model_type': model_type,
            }
        }
    ]

    val_dataloader = SizedNNCFDataLoaderImpl(validation_dataset,
                                             validation_dataset._transform_fn,
                                             validation_dataset.batch_size)

    engine = CustomEngine(engine_config, calibration_dataset, val_dataloader, validation_fn, higher_better)
    pipeline = pot.create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(pot_model)
    pot.compress_model_weights(compressed_model)

    compressed_model_paths = pot.save_model(compressed_model, save_path='/tmp/pot', model_name='model')
    ir_model_xml = compressed_model_paths[0]['model']
    ir_model_bin = compressed_model_paths[0]['weights']
    ie = openvino.runtime.Core()
    quantized_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    return quantized_model
