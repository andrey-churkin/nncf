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

from typing import Callable

from collections.abc import Sized

from nncf.ptq.api.types import DataSource
from nncf.ptq.api.types import DataItem
from nncf.ptq.api.types import ModelInput
from nncf.ptq.data.dataloader import NNCFDataLoaderImpl


# TODO(andrey-churkin): The algorithms from the POT use the `__len__()` method.
# It should be removed when we change all algorithms.
class SizedNNCFDataLoaderImpl(Sized, NNCFDataLoaderImpl):
    """
    Adds the `__len__()` method to the `NNCFDataLoaderImpl`.
    """

    def __init__(self,
                 data_source: DataSource,
                 transform_fn: Callable[[DataItem], ModelInput],
                 batch_size: int):
        """
        Initializes the NNCF data loader.

        :param data_source: The custom data source that is an iterable
            python object.
        :param transform_fn: Transformation method that takes a data item
            returned per iteration through `data_source` object and transforms
            it to the model's expected input that can be used for the model inference.
        :param batch_size: An integer that represents the number of consecutive elements
            of `data_source` that were combined in a single batch.
        """
        super().__init__(data_source, transform_fn, batch_size)
        self._length = None

    def __len__(self) -> int:
        if self._length is None:
            self._length = _get_length(self._data_source)
        return self._length


def _get_length(data_source: DataSource) -> int:
    """
    Returns the length of the provided custom data source.

    :param data_source: Custom data source that is an iterable python object.
    :return: The length of the provided custom data source.
    """
    if isinstance(data_source, Sized):
        return len(data_source)

    length = 0
    for _ in data_source:
        length = length + 1

    return length
