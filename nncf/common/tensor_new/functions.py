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

import functools
from typing import Optional, Tuple, Union, TypeVar


from nncf.common.tensor_new.enums import TensorDeviceType
from nncf.common.tensor_new.enums import TensorDataType


TTensor = TypeVar("TTensor")
T = TypeVar("T") # TODO: Verify


@functools.singledispatch
def device(a: TTensor) -> TensorDeviceType:
    """
    :param a:
    :return:
    """


@functools.singledispatch
def squeeze(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:
    """
    :param a:
    :param axis:
    :return:
    """


@functools.singledispatch
def flatten(a: TTensor) -> TTensor:
    """
    :param a:
    :return:
    """


@functools.singledispatch
def max(a: TTensor, axis: Optional[T] = None) -> TTensor:
    """
    :param a:
    :param axis:
    :return:
    """


@functools.singledispatch
def min(a: TTensor, axis: Optional[T] = None) -> TTensor:
    """
    :param a:
    :param axis:
    :return:
    """


@functools.singledispatch
def abs(a: TTensor) -> TTensor:
    """
    :param a:
    :return:
    """


@functools.singledispatch
def as_type(a: TTensor, dtype: TensorDataType):
    """
    :param a:
    :param dtype:
    """


@functools.singledispatch
def reshape(a: TTensor, shape: T) -> TTensor:
    """
    :param a:
    :param shape:
    :return:
    """


###############################################################################


@functools.singledispatch
def all(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or tuple of axes along which to count non-zeros. Default is None,
       meaning that non-zeros will be counted along a flattened version of a.
    :return: A new boolean or tensor is returned unless out is specified,
      in which case a reference to out is returned.
    """


@functools.singledispatch
def allclose(a: TTensor, b: TTensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> TTensor:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: True if the two arrays are equal within the given tolerance, otherwise False.
    """


@functools.singledispatch
def any(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or tuple of axes along which to count non-zeros. Default is None,
       meaning that non-zeros will be counted along a flattened version of a.
    :return: A new boolean or tensor is returned unless out is specified,
      in which case a reference to out is returned.
    """


@functools.singledispatch
def count_nonzero(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:
    """
    Counts the number of non-zero values in the tensor input.

    :param a: The tensor for which to count non-zeros.
    :param axis: Axis or tuple of axes along which to count non-zeros. Default is None,
       meaning that non-zeros will be counted along a flattened version of a.
    :return: Number of non-zero values in the tensor along a given axis.
      Otherwise, the total number of non-zero values in the tensor is returned.
    """


@functools.singledispatch
def is_empty(a: TTensor) -> bool:
    """
    Return True if input tensor is empty.

    :param a: The input tensor.
    :return: True is tensor is empty, otherwise False.
    """


@functools.singledispatch
def isclose(a: TTensor, b: TTensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> TTensor:
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: Returns a boolean tensor of where a and b are equal within the given tolerance.
    """


@functools.singledispatch
def maximum(x1: TTensor, x2: TTensor) -> TTensor:
    """
    Element-wise maximum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """


@functools.singledispatch
def minimum(x1: TTensor, x2: TTensor) -> TTensor:
    """
    Element-wise minimum of tensor elements.

    :param input: The first input tensor.
    :param other: The second input tensor.
    :return: Output tensor.
    """


@functools.singledispatch
def ones_like(a: TTensor) -> TTensor:
    """
    Return an tensor of ones with the same shape and type as a given tensor.

    :param a: The shape and data-type of a define these same attributes of the returned tensor.
    :return: Tensor of ones with the same shape and type as a.
    """


@functools.singledispatch
def where(condition: TTensor, x: TTensor, y: TTensor) -> TTensor:
    """
    Return elements chosen from x or y depending on condition.

    :param condition: Where True, yield x, otherwise yield y.
    :param x: Value at indices where condition is True.
    :param y: Value at indices where condition is False.
    :return: An tensor with elements from x where condition is True, and elements from y elsewhere.
    """


@functools.singledispatch
def zeros_like(a: TTensor) -> TTensor:
    """
    Return an tensor of zeros with the same shape and type as a given tensor.

    :param input: The shape and data-type of a define these same attributes of the returned tensor.
    :return: tensor of zeros with the same shape and type as a.
    """
