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

import os
import argparse
import subprocess
import time
from typing import Tuple

import openvino
from openvino.runtime import Core
import torch
import torchvision
import numpy as np

from nncf import ptq
from nncf.common.utils.logger import logger as nncf_logger
from nncf.ptq.examples.mobilenet_v2 import usercode


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.exists(args.example_dir):
        os.makedirs(args.example_dir)

    # Step 1: Prepare OpenVINO model.
    ir_model_xml, ir_model_bin = prepare_openvino_model(args.example_dir)
    ie = Core()
    original_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    # Step 2: Create dataset.
    data_source = create_val_dataset(args.dataset_dir)

    # Step 3: Apply quantization algorithm.
    quantized_model = quantize_model(original_model, data_source)

    # Step 4: Save quantized model.
    model_name = 'mobilenet_v2_quantized'
    ir_qmodel_xml = os.path.join(args.example_dir, f'{model_name}.xml')
    ir_qmodel_bin = os.path.join(args.example_dir, f'{model_name}.bin')
    openvino.offline_transformations.serialize(quantized_model, ir_qmodel_xml, ir_qmodel_bin)

    # Step 5: Compare the accuracy of the original and quantized models
    nncf_logger.info('Checking the accuracy of the original model:\n')
    original_compiled_model = ie.compile_model(original_model, device_name='CPU')
    validate(data_source, original_compiled_model, args.print_freq)

    nncf_logger.info('Checking the accuracy of the quantized model:\n')
    quantized_compiled_model = ie.compile_model(quantized_model, device_name='CPU')
    validate(data_source, quantized_compiled_model, args.print_freq)

    # Step 6: Compare Performance of the original and quantized models
    # Commands:
    # benchmark_app -m mobilenet_v2_quantization/mobilenet_v2.xml -d CPU -api async
    # benchmark_app -m mobilenet_v2_quantization/mobilenet_v2_quantized.xml -d CPU -api async


def create_parser():
    """
    Creates argument parser for the app.

    :return: The `ArgumentParser` object.
    """
    parser = argparse.ArgumentParser(description='Quantization of the MobileNetV2 model')
    parser.add_argument(
        '--example_dir',
        metavar='DIR',
        nargs='?',
        default='mobilenet_v2_quantization',
        help='path to a directory where the quantized model will be saved (default: mobilenet_v2_quantization)'
    )
    parser.add_argument(
        '--dataset_dir',
        metavar='DIR',
        nargs='?',
        default='imagenet',
        help='path to dataset (default: imagenet)'
    )
    parser.add_argument(
        '--print_freq',
        metavar='N',
        default=10000,
        type=int,
        help='print frequency (default: 10000)'
    )

    return parser


def prepare_openvino_model(example_dir: str) -> Tuple[str, str]:
    """
    Converts PyTorch MobileNetV2 model to the OpenVINO IR format.

    :param example_dir: Directory where OpenVINO IR model will be saved.
    :return: A tuple (ir_model_xml, ir_model_bin) where
        `ir_model_xml` - path to .xml file.
        `ir_model_bin` - path to .bin file.
    """
    # Step 1: Initialize model from the PyTorch Hub.
    # For more details, please see the [link](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    model_name = 'mobilenet_v2'
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    model.eval()

    # Step 2: Export PyTorch model to ONNX format.
    onnx_model_path = os.path.join(example_dir, f'{model_name}.onnx')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    # Step 3: Run Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = (
        'mo '
        '--framework onnx '
        '--data_type FP16 '
        '--input_shape [1,3,224,224] '
        f'-m {onnx_model_path} '
        f'--output_dir {example_dir}'
    )
    subprocess.call(mo_command, shell=True)

    # Step 4: Return path to IR model as result.
    ir_model_xml = os.path.join(example_dir, f'{model_name}.xml')
    ir_model_bin = os.path.join(example_dir, f'{model_name}.bin')
    return ir_model_xml, ir_model_bin


def create_val_dataset(dataset_dir: str) -> torch.utils.data.Dataset:
    """
    Creates validation ImageNet dataset.

    :param dataset_dir: Path to directory where ImageNet dataset is located.
    :return: The `torch.utils.data.Dataset` object.
    """
    val_dir = os.path.join(dataset_dir, 'val')
    # Transformations were taken from [here](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, preprocess)

    return val_dataset


def quantize_model(model: openvino.runtime.Model,
                   data_source: torch.utils.data.Dataset) -> openvino.runtime.Model:
    """
    Applies quantization algorithm to the OpenVINO model.

    :param model: An OpenVINO model to be quantized.
    :param data_source: A framework-specific data source for the calibration process.
    :return: Quantized OpenVINO model.
    """
    # Step 1: Define transformation method. This method should
    # take a data item from the data source and transform it
    # into the model expected input.
    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    # Step 2: Wrap framework-specific data source to the `NNCFDataLoader` object.
    calibration_dataset = ptq.make_dataloader(data_source, transform_fn)

    # Step 3: Apply quantization algorithm.
    quantized_model = ptq.quantize(model, calibration_dataset)

    return quantized_model


# This method was taken as is from the pythorch repository.
# You can find it [here](https://github.com/pytorch/examples/blob/main/imagenet/main.py).
# Code regarding CUDA and training was commented out.
# We put such code in the `NOT_NEEDED` block.
# Some code was changed and added. We put such code in the `BEFORE`` and `AFTER` blocks.

# BEFORE {
# def validate(val_loader, model, criterion, args):
# }

# AFTER {
def validate(val_loader, model, print_freq: int):
    output_layer = next(iter(model.outputs))
# }
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                # NOT_NEEDED {
                # if args.gpu is not None:
                #     images = images.cuda(args.gpu, non_blocking=True)
                # if torch.cuda.is_available():
                #     target = target.cuda(args.gpu, non_blocking=True)
                # }

                # compute output

                # BEFORE {
                # output = model(images)
                # }

                # AFTER {
                target = torch.from_numpy(np.expand_dims(np.array([target]), 0))
                input_data = np.expand_dims(images.numpy(), 0).astype(np.float32)
                output = torch.from_numpy(
                    model([input_data])[output_layer]
                )
                # }

                # NOT_NEEDED {
                # loss = criterion(output, target)
                # }

                # measure accuracy and record loss
                acc1, acc5 = usercode.accuracy(output, target, topk=(1, 5))

                # NOT_NEEDED {
                # losses.update(loss.item(), images.size(0))
                # }

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i + 1)

    batch_time = usercode.AverageMeter('Time', ':6.3f', usercode.Summary.NONE)

    # NOT_NEEDED {
    # losses = usercode.AverageMeter('Loss', ':.4e', usercode.Summary.NONE)
    # }

    top1 = usercode.AverageMeter('Acc@1', ':6.2f', usercode.Summary.AVERAGE)
    top5 = usercode.AverageMeter('Acc@5', ':6.2f', usercode.Summary.AVERAGE)

    # BEFORE {
    # progress = usercode.ProgressMeter(
    #     len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    #     [batch_time, losses, top1, top5],
    #     prefix='Test: ')
    # }
    # AFTER {
    progress = usercode.ProgressMeter(
        len(val_loader), [batch_time, top1, top5], prefix='Test: '
    )
    # }

    # NOT_NEEDED {
    # switch to evaluate mode
    # model.eval()
    # }

    run_validate(val_loader)

    # NOT_NEEDED {
    # if args.distributed:
    #     top1.all_reduce()
    #     top5.all_reduce()
    #
    # if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
    #     aux_val_dataset = Subset(val_loader.dataset,
    #                              range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    #     aux_val_loader = torch.utils.data.DataLoader(
    #         aux_val_dataset, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
    #     run_validate(aux_val_loader, len(val_loader))
    # }

    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    run_example()
