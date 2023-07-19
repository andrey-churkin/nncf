# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Iterator, List, Optional, Tuple, TypeVar, Union
from nncf.common.tensor_new.enums import TensorDataType
from nncf.common.tensor_new.enums import TensorDeviceType
from nncf.common.tensor_new import functions


DataType = TypeVar("DataType")


def _initialize_backends():
    try:
        from nncf.common.tensor_new import numpy_functions
    except ImportError:
        pass

    try:
        from nncf.common.tensor_new import torch_functions
    except ImportError:
        pass


_initialize_backends()


class Tensor:
    """
    An interface to framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, data: Optional[DataType]):
        self._data = data.data if isinstance(data, Tensor) else data

    @property
    def data(self) -> DataType:
        return self._data

    @property
    def shape(self) -> List[int]:
        return Tensor(list(self.data.shape))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __getitem__(self, index: int) -> "Tensor":
        return Tensor(self.data[index])

    # built-in operations

    def __add__(self, other: DataType) -> "Tensor":
        return Tensor(self.data + unwrap_tensor_data(other))

    def __radd__(self, other: DataType) -> "Tensor":
        return Tensor(unwrap_tensor_data(other) + self.data)

    def __sub__(self, other: DataType) -> "Tensor":
        return Tensor(self.data - unwrap_tensor_data(other))

    def __rsub__(self, other: DataType) -> "Tensor":
        return Tensor(unwrap_tensor_data(other) - self.data)

    def __mul__(self, other: DataType) -> "Tensor":
        return Tensor(self.data * unwrap_tensor_data(other))

    def __rmul__(self, other: DataType) -> "Tensor":
        return Tensor(unwrap_tensor_data(other) * self.data)

    def __pow__(self, other: DataType) -> "Tensor":
        return Tensor(self.data ** unwrap_tensor_data(other))

    def __truediv__(self, other: DataType) -> "Tensor":
        return Tensor(self.data / unwrap_tensor_data(other))

    def __floordiv__(self, other: DataType) -> "Tensor":
        return Tensor(self.data // unwrap_tensor_data(other))

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    # Comparison operators

    def __lt__(self, other: DataType) -> "Tensor":
        return Tensor(self.data < unwrap_tensor_data(other))

    def __le__(self, other: DataType) -> "Tensor":
        return Tensor(self.data <= unwrap_tensor_data(other))

    def __eq__(self, other: DataType) -> "Tensor":
        return Tensor(self.data == unwrap_tensor_data(other))

    def __nq__(self, other: DataType) -> "Tensor":
        return Tensor(self.data != unwrap_tensor_data(other))

    def __gt__(self, other: DataType) -> "Tensor":
        return Tensor(self.data > unwrap_tensor_data(other))

    def __ge__(self, other: DataType) -> "Tensor":
        return Tensor(self.data >= unwrap_tensor_data(other))

    # Tensor functions

    @property
    def device(self) -> TensorDeviceType:
        return functions.device(self.data)

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]] = None) -> "Tensor":
        return Tensor(functions.squeeze(self.data, axis))

    def flatten(self) -> "Tensor":
        return Tensor(functions.flatten(self.data))

    def max(self, axis: Optional[DataType] = None) -> "Tensor":
        return Tensor(functions.max(self.data, axis))

    def min(self, axis: Optional[DataType] = None) -> "Tensor":
        return Tensor(functions.min(self.data, axis))

    def abs(self) -> "Tensor":
        return Tensor(functions.abs(self.data))

    def is_empty(self) -> "Tensor":
        return Tensor(functions.is_empty(self.data))

    def as_type(self, dtype: TensorDataType):
        return Tensor(functions.as_type(self.data, dtype))

    def reshape(self, shape: DataType) -> "Tensor":
        return Tensor(functions.reshape(self.data, shape))


def unwrap_tensor_data(obj: Any) -> DataType:
    """
    Return the data of a Tensor object, or the object itself if it is not a Tensor.

    :param obj: The object to unwrap.
    :return: The data of the Tensor object, or the object itself.
    """
    return obj.data if isinstance(obj, Tensor) else obj
