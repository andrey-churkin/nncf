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

from typing import List
from typing import Iterator

from nncf.ptq.api.types import DataItem
from nncf.ptq.api.dataloader import NNCFDataLoader
from nncf.ptq.data.dataloader import NNCFDataLoaderImpl


def create_subset(data_loader: NNCFDataLoader, indices: List[int]) -> NNCFDataLoader:
    """
    Create a new instance of `NNCFDataLoader` that contains only
    specified batches.

    :param data_loader: The data loader to select the specified elements.
    :param indices: The zero-based indices of batches that should be
        selected from provided data loader. The indices should be sorted
        in ascending order.
    :return: The new instance of `NNCFDataLoader` that contains only
        specified batches.
    """
    class BatchSelector:
        def __iter__(self) -> Iterator[DataItem]:
            pos = 0  # Position in the `indices` list.
            num_indices = len(indices)

            for idx, batch in enumerate(data_loader):
                if pos == num_indices:
                    # All specified batches were selected.
                    break
                if idx == indices[pos]:
                    pos = pos + 1
                    yield batch

    return NNCFDataLoaderImpl(BatchSelector(), data_loader._transform_fn, data_loader.batch_size)
