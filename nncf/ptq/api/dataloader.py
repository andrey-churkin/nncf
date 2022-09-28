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

from typing import Iterator
from abc import ABC
from abc import abstractmethod

from nncf.ptq.api.types import DataItem
from nncf.ptq.api.types import ModelInput


class NNCFDataLoader(ABC):
    """
    Describes the interface of the data source that is used by
    compression algorithms.

    The `NNCFDataLoader` object contains the dataset and information
    about how to transform the data item returned per iteration to
    the model's expected input.
    """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        Returns the number of elements return per iteration.

        :return: A number of elements return per iteration.
        """

    @abstractmethod
    def transform(self, data: DataItem) -> ModelInput:
        """
        Transforms the passed argument to the model input.

        :param data: The data element returned per iteration through
            this data loader.
        :return: Model's expected input that can be used for the model
            inference.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[DataItem]:
        """
        Creates an iterator for the data items of this data loader.

        :return: An iterator for the data items of this data loader.
        """
