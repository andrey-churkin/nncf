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

import openvino.runtime as ov

from abc import ABC, abstractmethod
from typing import Iterable, Any, List, Generic, Hashable, Optional, TypeVar
from collections import OrderedDict


T = TypeVar('T')


class LruCache(Generic[T]):
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._cache = OrderedDict()  # type: OrderedDict[Hashable, T]

    @property
    def capacity(self) -> int:
        return self._capacity

    def get(self, key: Hashable) -> Optional[T]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Hashable, value: T) -> None:
        if len(self._cache) == self._capacity:
            self._cache.popitem(last=False)

        self._cache[key] = value
        self._cache.move_to_end(key)

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


class Evaluator(ABC):

    def __init__(self):
        self._lru_cache = LruCache(capacity=2)
        self._model_lru_cache = LruCache(capacity=2)
        self._values_appending = True
        self._metrics_for_each_item = []
        self._logits_for_each_item = []

    # API for internal use only

    def __call__(self, model: ov.Model, dataset: Iterable[Any]) -> float:
        compiled_model = self._prepare_model_for_inference(model)

        metric = self.validate(compiled_model, dataset)
        if self.is_values_appending():
            self._check()
            if self._metrics_for_each_item:
                self._lru_cache.put(model, self._metrics_for_each_item)
            else:  # self._logits_for_each_item is not empty
                self._lru_cache.put(model, self._logits_for_each_item)
        return metric

    def _check(self):
        append_metric_for_single_item_was_called = False
        append_logits_for_single_item_was_called = False

        if self._metrics_for_each_item:
            append_metric_for_single_item_was_called = True
        if self._logits_for_each_item:
            append_logits_for_single_item_was_called = True

        if append_metric_for_single_item_was_called and append_logits_for_single_item_was_called:
            raise RuntimeError('Only one method from '
                               'append_metric_for_single_item() and append_logits_for_single_item() '
                               'should be called inside the validate() method.')
        if not (append_metric_for_single_item_was_called or append_logits_for_single_item_was_called):
            raise RuntimeError('One method from '
                               'append_metric_for_single_item() and append_logits_for_single_item() '
                               'should be called inside the validate() method.')

    def _prepare_model_for_inference(self, model: ov.Model) -> ov.CompiledModel:
        compiled_model = self._model_lru_cache.get(model)
        if compiled_model is None:
            compiled_model = ov.compile_model(model)
            self._model_lru_cache.put(model, compiled_model)
        return compiled_model

    def is_values_appending(self) -> bool:
        return self._values_appending

    def enable_values_appending(self) -> None:
        self._values_appending = True

    def disable_values_appending(self) -> None:
        self._values_appending = False

    def get_metrics_for_each_item(self, model: ov.Model) -> Optional[List[float]]:
        return self._lru_cache.get(model)

    def get_logits_for_each_item(self, model: ov.Model) -> Optional[List[Any]]:
        return self._lru_cache.get(model)

    # API that should be used by user:

    @abstractmethod
    def validate(self, compiled_model: ov.CompiledModel, dataset: Iterable[Any]) -> float:
        pass

    def append_metric_for_single_item(self, metric_value: float) -> None:
        if self.is_values_appending():
            self._metrics_for_each_item.append(metric_value)

    def append_logits_for_single_item(self, logits: Any) -> None:
        if self.is_values_appending():
            self._logits_for_each_item.append(logits)
