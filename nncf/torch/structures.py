"""
 Copyright (c) 2020 Intel Corporation
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
from typing import Callable, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.config.structures import NNCFExtraConfigStruct


class QuantizationPrecisionInitArgs(NNCFExtraConfigStruct):
    """
    Stores arguments for initialization of quantization's bitwidth.
    Initialization is based on calculating a measure reflecting layers' sensitivity to perturbations. The measure is
    calculated by estimation of average trace of Hessian for modules using the Hutchinson algorithm.
    :param criterion_fn: callable object, that implements calculation of loss by given outputs of the model, targets,
    and loss function. It's not needed when the calculation of loss is just a direct call of the criterion with 2
    arguments: outputs of model and targets. For all other specific cases, the callable object should be provided.
    E.g. for inception-v3, the losses for two outputs of the model are combined with different weight.
    :param criterion: loss function, instance of descendant of `torch.nn.modules.loss._Loss`,
    :param data_loader: 'data_loader' - provides an iterable over the given dataset. Instance of
                nncf.initialization.PTInitializingDataLoader; a regular 'torch.utils.data.DataLoader' may
                also be passed, but only in the simple case when it returns a tuple of (input, target) tensors.
                *WARNING*: The final quantizer setup of the created compressed model is dependent on the data
                provided by the data_loader. When using PyTorch's DistributedDataParallel with precision
                initialization, make sure that each process in the distributed group receives the same data
                from the data_loader as the other processes, otherwise the create_compressed_model call may
                create different compressed model objects for each distributed process and the distributed training
                will fail.
    :param device: Device to perform initialization at. Either 'cpu', 'cuda', or None (default); if None, will
                   use the device of the model's parameters.
    """

    def __init__(self, criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor], criterion: _Loss,
                 data_loader: DataLoader, device: str = None):
        self.criterion_fn = criterion_fn
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device

    @classmethod
    def get_id(cls) -> str:
        return "quantization_precision_init_args"


class AutoQPrecisionInitArgs(NNCFExtraConfigStruct):
    """
    :param data_loader: 'data_loader' - provides an iterable over the given dataset. Instance of
                nncf.initialization.PTInitializingDataLoader; a regular 'torch.utils.data.DataLoader' may
                also be passed, but only in the simple case when it returns a tuple of (input, target) tensors.
                *WARNING*: The final quantizer setup of the created compressed model is dependent on the data
                provided by the data_loader. When using PyTorch's DistributedDataParallel with precision
                initialization, make sure that each process in the distributed group receives the same data
                from the data_loader as the other processes, otherwise the create_compressed_model call may
                create different compressed model objects for each distributed process and the distributed training
                will fail.
    """
    def __init__(self, data_loader: DataLoader,
                 eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float],
                 nncf_config: 'NNCFConfig'):
        self.data_loader = data_loader
        self.eval_fn = eval_fn
        self.config = nncf_config

    @classmethod
    def get_id(cls) -> str:
        return "autoq_precision_init_args"


class LeGRInitArgs(NNCFExtraConfigStruct):
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 train_fn: Callable[[torch.utils.data.DataLoader, torch.nn.Module,
                                     torch.optim.Optimizer, 'CompressionAlgorithmController',
                                     Optional[int]], type(None)],
                 val_loader: torch.utils.data.DataLoader,
                 val_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader],
                                  Tuple[float, float, float]],
                 train_optimizer: torch.optim.Optimizer,
                 nncf_config: 'NNCFConfig'):
        self.train_loader = train_loader
        self.train_steps_fn = train_fn
        self.val_loader = val_loader
        self.val_fn = val_fn
        self.train_optimizer = train_optimizer
        self.config = nncf_config

    @classmethod
    def get_id(cls) -> str:
        return "legr_init_args"


class DistributedCallbacksArgs(NNCFExtraConfigStruct):
    def __init__(self,
                 wrapping_callback: Callable[[nn.Module], nn.Module],
                 unwrapping_callback: Callable[[nn.Module], nn.Module]):
        """
        Pair of callbacks that needed for distributed training of the model: wrapping model with wrapping_callback for
        distributed training, and after all training steps unwrapping model to the initial not-distributed state with
        unwrapping_callback.
        :param wrapping_callback: Callback that wraps model for distributed training with any necessary structure (for
        example, torch.nn.DataParallel or any custom class), returns wrapped model ready for distributed training
        :param unwrapping_callback: Callback for unwrapping model wrapped with wrapping_callback, returns original model
        """
        self.wrap_model = wrapping_callback
        self.unwrap_model = unwrapping_callback

    @classmethod
    def get_id(cls) -> str:
        return "distributed_callbacks_args"


class ExecutionParameters:
    def __init__(self, cpu_only: bool, current_gpu: Optional[int]):
        """
        Parameters that is necessary for distributed training of the model.
        :param cpu_only: whether cpu-only mode is using for training
        :param current_gpu: id of GPU that should be used for training (if only one of all is used)
        """
        self.cpu_only = cpu_only
        self.current_gpu = current_gpu
