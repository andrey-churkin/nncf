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

from typing import Dict
from typing import Any
import threading


_CURRENT_CONTEXT = threading.local()


def get_current_context():
    """
    Returns current active `TFTracingContext`.

    :return: Tracing context.
    """
    tracing_context = getattr(_CURRENT_CONTEXT, 'tracing_context', None)
    if tracing_context is None:
        tracing_context = TFTracingContext()
        setattr(_CURRENT_CONTEXT, 'tracing_context', tracing_context)
    return tracing_context


class TFTracingContext:
    """
    Contains information about should we wrap the TensorFlow
    operation or not.
    """

    def __init__(self):
        """
        Initializes the tracing context.
            - in_call: `True` if we are inside the `call` method of the
                tf.keras.Model instance, `False` otherwise.
            - wrap_ops: `True` if we should wrap the TensorFlow operation,
                `False` otherwise.
        """
        self._state = {
            'in_call': False,
            'wrap_ops': False,
        }
        # Maps a name used in the graph to the next id to use for that name.
        self.names_in_use = {}
        self.model = None

    @property
    def in_call(self) -> bool:
        return self._state['in_call']

    @property
    def wrap_ops(self) -> bool:
        return self._state['wrap_ops']

    def enter(self, wrap_ops: bool, in_call: bool = True):
        next_state = {
            'wrap_ops': wrap_ops,
            'in_call': in_call,
        }
        return TFTracingContextManager(self, next_state)

    def unique_name(self, name: str) -> str:
        """
        Returns a unique operation name for `name`.

        For more details, please, see implementation of
        the `tf.Graph.unique_name()` method.

        :param name: The name for an operation.
        :return: Unique name.
        """
        name_key = name.lower()

        i = self.names_in_use.get(name_key, 0)
        self.names_in_use[name_key] = i + 1

        if i > 0:
            base_name_key = name_key
            # Make sure the composed name key is not already used.
            while name_key in self.names_in_use:
                name_key = f'{base_name_key}_{i}'
                i += 1

            # Mark the composed name_key as used in case someone wants
            # to call unique_name('name_1').
            self.names_in_use[name_key] = 1

            # Return the new name with the original capitalization of the given name.
            name = f'{name}_{i - 1}'
        return name

    def get_state(self) -> Dict[str, Any]:
        return self._state

    def load_state(self, state: Dict[str, Any]) -> None:
        self._state = state


class TFTracingContextManager:
    """
    Context manager for the tracing context.
    """

    def __init__(self,
                 tracing_context: TFTracingContext,
                 next_state: Dict[str, Any]):
        """
        Initializes the tracing context manager.

        :param tracing_context: Tracing context.
        :param next_state: Next state of the tracing context which
            should be applied.
        """
        self._tracing_context = tracing_context
        self._next_state = next_state
        self._prev_state = None

    def __enter__(self):
        self._prev_state = self._tracing_context.get_state()
        self._tracing_context.load_state(self._next_state)

    def __exit__(self, *exc):
        self._tracing_context.load_state(self._prev_state)

        if not self._tracing_context.in_call:
            self._tracing_context.names_in_use = {}