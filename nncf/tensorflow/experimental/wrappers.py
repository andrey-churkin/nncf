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

import functools
from typing import Callable, List

from nncf.tensorflow.layers.operation import NNCFOperation

from nncf.tensorflow.experimental.argprovider import TF_ARG_PROVIDERS
from nncf.tensorflow.experimental.nncf_context import get_nncf_context
from nncf.tensorflow.experimental.scope import get_op_name


class TFHook:
    """
    Contains the NNCF operation and information to apply it.
    """

    def __init__(self,
                 nncf_operation: NNCFOperation,
                 port_id: int,
                 is_pre_hook: bool,
                 op_type_name: str,
                 priority: int):
        """
        Initializes a `TFHook`.

        :param nncf_operation: A NNCF operation.
        :param port_id: A zero-based index of input for which
            `nncf_operation` should be applied.
        :param is_pre_hook: `True` for pre-hook, `False` for post-hook.
        :param op_type_name: Type of TensorFlow operation (see the tf.raw_ops module)
            for which hook is applied.
        :param priority: Priority of hook.
        """
        self._nncf_operation = nncf_operation
        self._port_id = port_id
        self._is_pre_hook = is_pre_hook
        self._op_type_name = op_type_name
        self._priority = priority

        self._arg_provider = TF_ARG_PROVIDERS.get(self._op_type_name)()
        if self._arg_provider is None:
            raise ValueError(f'Unexpected type of TensorFlow operation: {self._op_type_name}. '
                             'Register an `ArgProvider` instance for this type in the '
                             '`TF_ARG_PROVIDERS` registry, please.')

    @property
    def is_pre_hook(self) -> bool:
        return self._is_pre_hook

    @property
    def priority(self) -> int:
        return self._priority

    def __call__(self, *args, **kwargs):
        """
        Applies this hook.

        :return: A tuple (args, kwargs).
        """

        if self._is_pre_hook:
            get_fn = self._arg_provider.get_input
            set_fn = self._arg_provider.set_input
        else:
            get_fn = self._arg_provider.get_output
            set_fn = self._arg_provider.set_output

        x = get_fn(self._port_id, args, kwargs)
        y = self._nncf_operation(*(x,))
        args_, kwargs_ = set_fn(self._port_id, y, args, kwargs)

        return args_, kwargs_


def _apply_hooks(hooks: List[TFHook], *args, **kwargs):
    """
    Applies hooks.

    :param hooks: A list of hooks which sorted by priority.
    :return: A tuple (args, kwargs).
    """
    for hook in hooks:
        args, kwargs = hook(*args, **kwargs)
    return args, kwargs


def wrap_tf_operation(op: Callable,
                      op_type_name: str,
                      pre_hooks: List[TFHook],
                      post_hooks: List[TFHook]) -> Callable:
    """
    Wraps TensorFlow op.

    :param op: TensorFlow op i.e. function from the `tf.raw_ops` module.
    :param op_type_name: Operation type name (name of the function
        from the `tf.raw_ops` module)
    :param pre_hooks:
    :param post_hooks:
    :return: Wrapped function.
    """
    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        nncf_context = get_nncf_context()

        op_name = get_op_name(op_type_name, kwargs.get('name'))

        # Should we wrap current operation?
        if not nncf_context.wrap_ops:
            return op(*args, **kwargs)

        with nncf_context.enter(wrap_ops=False):
            # Apply pre-hooks
            args, kwargs = _apply_hooks(pre_hooks.get(op_name, []), *args, **kwargs)

            # Apply TensorFlow operation
            outputs = op(*args, **kwargs)

            # Apply post-hooks
            (outputs,), _ = _apply_hooks(post_hooks.get(op_name, []), *(outputs,), **{})

        return outputs

    return wrapper
