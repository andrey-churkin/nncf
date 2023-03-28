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

from pathlib import Path
from typing import Iterable, Any

import numpy as np
import openvino.runtime as ov
import torch
from fastdownload import FastDownload
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
MODEL_URL = 'https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/openvino_model.tgz'
MODEL_PATH = '~/.cache/nncf/models'
DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
DATASET_PATH = '~/.cache/nncf/datasets'
DATASET_CLASSES = 10


def download(url: str, path: str) -> Path:
    downloader = FastDownload(base=path,
                              archive='downloaded',
                              data='extracted')
    return downloader.get(url)

# Before:
def validate(model: ov.Model,
             val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    compiled_model = ov.compile_model(model)
    output = compiled_model.outputs[0]

    for images, target in tqdm(val_loader):
        pred = compiled_model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)

# After:
from examples.post_training_quantization.openvino.mobilenet_v2.evaluator import Evaluator

class MobileNetV2Evaluator(Evaluator):  # MobileNetV2Evaluator(nncf.Evaluator)
    """
    Evaluator for the MobileNetV2 model.
    """

    def validate(self, compiled_model: ov.CompiledModel, dataset: Iterable[Any]) -> float:
        predictions = []
        references = []
        output = compiled_model.outputs[0]

        for images, target in tqdm(dataset):
            pred = compiled_model(images)[output]

            # self.append_logits_for_single_item(pred)  # For logits

            pred = np.argmax(pred, axis=1)

            self.append_metric_for_single_item(accuracy_score(pred, target))  # For metric

            predictions.append(pred)
            references.append(target)

        predictions = np.concatenate(predictions, axis=0)
        references = np.concatenate(references, axis=0)
        return accuracy_score(predictions, references)


###############################################################################
# Create an OpenVINO model and dataset

dataset_path = download(DATASET_URL, DATASET_PATH)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    root=f'{dataset_path}/val',
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False)

model_path = download(MODEL_URL, MODEL_PATH)
model = ov.Core().read_model(model_path / 'mobilenet_v2_fp32.xml')

###############################################################################
evaluator = MobileNetV2Evaluator()

fp32_top1 = evaluator(model, val_loader)
print(f'Accuracy @ top1: {fp32_top1:.3f}')

metrics_for_each_item = evaluator.get_metrics_for_each_item(model)
print('Metrics for each item:')
print(metrics_for_each_item[:10])
