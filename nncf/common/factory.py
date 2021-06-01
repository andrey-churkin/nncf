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

from typing import Optional, List, Tuple, Any

from nncf.api.compression import ModelType
from nncf.common.exporter import Exporter
from nncf.common.initialization import NNCFDataLoader
from nncf.common.utils.backend import __nncf_backend__


def create_exporter(model: ModelType,
                    input_names: Optional[List[str]] = None,
                    output_names: Optional[List[str]] = None,
                    model_args: Optional[Tuple[Any, ...]] = None) -> Exporter:
    """
    Factory for building an exporter.
    """
    if __nncf_backend__ == 'Torch':
        from nncf.torch.exporter import PTExporter
        exporter = PTExporter(model, input_names, output_names, model_args)
    elif __nncf_backend__ == 'TensorFlow':
        from beta.nncf.tensorflow.exporter import TFExporter
        exporter = TFExporter(model, input_names, output_names, model_args)

    return exporter


def create_bn_adaptation_algorithm_impl(data_loader: NNCFDataLoader,
                                        num_bn_adaptation_steps: int,
                                        num_bn_forget_steps: int,
                                        device: Optional[str] = None):
    """
    Factory for building a batchnorm adaptation algorithm implementation.

    :return: Implementation of the `BatchnormAdaptationAlgorithmImpl` class.
    """
    if __nncf_backend__ == 'Torch':
        from nncf.torch.batchnorm_adaptation import PTBatchnormAdaptationAlgorithmImpl
        bn_adaptation_algorithm_impl = PTBatchnormAdaptationAlgorithmImpl(data_loader,
                                                                          num_bn_adaptation_steps,
                                                                          num_bn_forget_steps,
                                                                          device)
    elif __nncf_backend__ == 'Tensorflow':
        from beta.nncf.tensorflow.batchnorm_adaptation import TFBatchnormAdaptationAlgorithmImpl
        bn_adaptation_algorithm_impl = TFBatchnormAdaptationAlgorithmImpl(data_loader,
                                                                          num_bn_adaptation_steps,
                                                                          num_bn_forget_steps,
                                                                          device)

    return bn_adaptation_algorithm_impl
