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

from nncf.compression_method_api import PTCompressionScheduler
from nncf.composite_compression import PTCompositeCompressionScheduler


class DummyScheduler(PTCompressionScheduler):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def state_dict(self):
        state = super().state_dict()
        state['delta'] = self.delta
        return state


def test_can_restore_from_state():
    def _create_composite_scheduler(deltas):
        schedulers = [DummyScheduler(delta) for delta in deltas]
        composite_scheduler = PTCompositeCompressionScheduler()
        for scheduler in schedulers:
            composite_scheduler.add(scheduler)
        return composite_scheduler

    expected = _create_composite_scheduler([1, 2])
    actual = _create_composite_scheduler([0, 0])
    actual.load_state_dict(expected.state_dict())

    for expected_child, actual_child in zip(expected.child_schedulers, actual.child_schedulers):
        assert expected_child.delta == actual_child.delta
