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

from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Type, TypeVar, Union

from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.tune_hyperparams.backend import ParamsGridSearchAlgoBackend
from nncf.quantization.algorithms.tune_hyperparams.params_transformation import ParamsTransformation
from nncf.quantization.algorithms.tune_hyperparams.params_transformation import create_params_transformation

SearchSpace = Dict[str, Union[List[Any], "SearchSpace"]]
TModel = TypeVar("TModel")


def get_algo_backend(backend: BackendType) -> ParamsGridSearchAlgoBackend:
    """
    Returns backend for params grid search algorithm.

    :param backend: Backend.
    :return: The backend for params grid search algorithm.
    """
    if backend == BackendType.OPENVINO:
        from nncf.quantization.algorithms.tune_hyperparams.openvino_backend import OVParamsGridSearchAlgoBackend

        return OVParamsGridSearchAlgoBackend()

    raise RuntimeError(
        f"Cannot create the backend for the params grid search algorithm because {backend} is not supported."
    )


class ParamsGridSearchAlgorithm:
    """ """

    def __init__(
        self,
        algorithm_cls: Type[Algorithm],
        init_params: Dict[str, Any],
        search_space: SearchSpace,
        subset_indices: List[int],
        validation_fn: Callable[[Any, Iterable[Any]], float],
    ):
        """
        :param algorithm_cls:
        :param init_params:
        :param search_space:
        :param subset_indices:
        :param validation_fn:
        """
        self._algorithm_cls = algorithm_cls
        self._init_params = init_params
        self._params_transformations = create_params_transformation(search_space)
        self._subset_indices = subset_indices
        self._validation_fn = validation_fn

    def apply(self, model: TModel, statistic_dataset: Dataset, validation_dataset: Dataset) -> TModel:
        """
        :param model:
        :param statistic_dataset:
        :param validation_dataset:
        """
        model_copy = copy_model(model)
        statistic_points = self._create_statistic_points(model_copy, statistic_dataset)
        algo_backend = get_algo_backend(get_backend(model_copy))

        best_transformation = None
        best_score = None

        for _, param_transformations in self._params_transformations.items():
            curr_best_transformation = best_transformation
            curr_best_score = best_score

            for curr_transformation in param_transformations:
                if curr_best_transformation:
                    curr_transformation = ParamsTransformation.concatenate(
                        curr_best_transformation, curr_transformation
                    )

                curr_params = curr_transformation.apply(self._init_params)
                algorithm = self._algorithm_cls(**curr_params)
                curr_model = algorithm.apply(model, statistic_points)
                curr_score = self._validation_fn(
                    algo_backend.prepare_for_inference(curr_model), validation_dataset.get_data(self._subset_indices)
                )

                if curr_score > curr_best_score:
                    curr_best_score = curr_score
                    curr_best_transformation = curr_transformation

            # Update global best transformation and score
            best_transformation = curr_best_transformation
            best_score = curr_best_score

        # Apply best parameters
        best_params = best_transformation.apply(self._init_params)
        algorithm = self._algorithm_cls(**best_params)
        best_model = algorithm.apply(model, statistic_points)

        return best_model

    def _create_statistic_points(self, model: TModel, dataset: Dataset) -> StatisticPointsContainer:
        stats_aggregator = StatisticsAggregatorFactory(model, dataset)

        for params in chain.from_iterable(self._params_transformations):
            algorithm = self._algorithm_cls(**params)
            stats_aggregator.register_statistic_points(algorithm.get_statistic_points(model))
        stats_aggregator.collect_statistics(model)

        return stats_aggregator.statistic_points


def quantize_with_tune_hyperparams(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], float],
    subset_indices: List[int],
    quantization_params: Dict[str, Any],
) -> TModel:
    """
    :param model:
    :param calibration_dataset:
    :param validation_dataset:
    :param validation_fn:
    :param subset_indices:
    :param quantization_params:
    """
    search_space = {
        "preset": [
            QuantizationPreset.PERFORMANCE,
            QuantizationPreset.MIXED,
        ],
        "fast_bias_correction": [
            True,
            False,
        ],
        # "advanced_parameters": {
        #     "weights_range_estimator_params": [
        #         RangeEstimatorParameters(
        #             min=StatisticsCollectorParameters(
        #             ),
        #             max=StatisticsCollectorParameters(
        #             )
        #         )
        #     ],
        #     "activations_range_estimator_params" : create_params(
        #         RangeEstimatorParameters,
        #         min=[
        #             StatisticsCollectorParameters(
        #             ),
        #             StatisticsCollectorParameters(
        #             ),
        #             StatisticsCollectorParameters(
        #             ),
        #         ],
        #         max=[
        #             StatisticsCollectorParameters(
        #             ),
        #             StatisticsCollectorParameters(
        #             ),
        #             StatisticsCollectorParameters(
        #             ),
        #         ]
        #     ),
        # },
    }

    algorithm = ParamsGridSearchAlgorithm(
        PostTrainingQuantization, quantization_params, search_space, subset_indices, validation_fn
    )
    quantized_model = algorithm.apply(model, calibration_dataset, validation_dataset)

    return quantized_model
