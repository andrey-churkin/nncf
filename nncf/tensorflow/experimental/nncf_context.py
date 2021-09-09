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

from typing import Dict, Any
import threading


_NNCF_CONTEXT = threading.local()


def get_nncf_context():
    """
    Returns current active `NNCFContext`.

    :return: NNCF context.
    """
    nncf_context = getattr(_NNCF_CONTEXT, 'nncf_context', None)
    if nncf_context is None:
        nncf_context = NNCFContext()
        setattr(_NNCF_CONTEXT, 'nncf_context', nncf_context)
    return nncf_context


class NNCFContext:
    """
    Contains information about should we wrap the TensorFlow
    operation or not.
    """

    def __init__(self):
        """
        Initializes the NNCF context.
            - in_call: `True` if we are inside the `call` method of the
                tf.keras.Model instance, `False` otherwise.
            - wrap_ops: `True` if we should wrap the TensorFlow operation,
                `False` otherwise.
        """
        self._state = {
            'in_call': False,
            'wrap_ops': False,
        }

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
        return NNCFContextManager(self, next_state)

    def get_state(self) -> Dict[str, Any]:
        return self._state

    def load_state(self, state: Dict[str, Any]) -> None:
        self._state = state


class NNCFContextManager:
    """
    Context manager for the NNCF context.
    """

    def __init__(self,
                 nncf_context: NNCFContext,
                 next_state: Dict[str, Any]):
        """
        Initializes the NNCF context.

        :param nncf_context: NNCF context.
        :param next_state: Next state of the NNCF context which
            should be applied.
        """
        self._nncf_context = nncf_context
        self._next_state = next_state
        self._prev_state = None

    def __enter__(self):
        self._prev_state = self._nncf_context.get_state()
        self._nncf_context.load_state(self._next_state)

    def __exit__(self, *exc):
        self._nncf_context.load_state(self._prev_state)
