from functools import reduce, partial

import torch
from torch import nn

from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.nested_objects_traversal import NestedObjectIndex


class KnowledgeDistillationLoss(PTCompressionLoss):
    """
    Doesn't not directly compute knowledge distillation loss values but has access to them through
    KnowledgeDistillationLossHandler. Notice that knowledge distillation loss is computed between results of original
    model and compressed model inferences with latest inputs. Provides KnowledgeDistillationLossHandler with kd original
    model (to distill from), storage device and function to calculate knowledge distillation loss.
    """
    def __init__(self, target_model: NNCFNetwork, original_model: nn.Module, kd_type: str):
        super().__init__()
        original_model.train()
        device = next(target_model.parameters()).device
        if kd_type == 'softmax':
            def kd_loss_fn(ref_outputs, compressed_model_outputs):
                return -(nn.functional.log_softmax(compressed_model_outputs, dim=1) *
                         nn.functional.softmax(ref_outputs, dim=1)).mean() * (compressed_model_outputs.shape[1])
        elif kd_type == 'mse':
            def kd_loss_fn(ref_outputs, compressed_model_outputs):
                mse = torch.nn.MSELoss()
                return mse(ref_outputs, compressed_model_outputs)
        else:
            raise ValueError('Choose between mse/softmax options for Knowledge Distillation')
        self._kd_loss_handler = target_model.create_knowledge_distillation_loss_handler(original_model, partial(
            KnowledgeDistillationLoss._calculate,
            device=device,
            kd_loss_fn=kd_loss_fn))

    @staticmethod
    def _calculate(compressed_model_outputs, orig_model_outputs, device: torch.device, kd_loss_fn) -> torch.Tensor:
        """
        Calculates knowledge distillation loss value from compressed_model_outputs and orig_model_outputs. First uses
        nested_object_paths_generator to unpack input containers and numerate contents inside them.
        Than checks compressed_model_outputs unpacked container for loss tensors (requires_grad=True)
        and maps extracted structure of loss tensors to orig_model_outputs.
        Finally computes knowledge distillation loss with extracted loss tensors.

        :param compressed_model_outputs: Output tensors of compressed model can be any type of container with
            deterministic traversal.
        :param orig_model_outputs: Output tensors of original model (used for distillation) can be any type of
            container with deterministic traversal.
        :return: knowledge distillation loss value
        """

        compressed_model_outputs_nested_obj_indexing = NestedObjectIndex([compressed_model_outputs])
        orig_model_outputs_nested_obj_indexing = NestedObjectIndex([orig_model_outputs])
        compressed_model_loss_outputs_nested_obj_indexing = list(filter(
            lambda x: KnowledgeDistillationLoss._is_loss(x.getter()),
            compressed_model_outputs_nested_obj_indexing.get_flat_nested_obj_indexing()))
        compressed_model_loss_outputs = list(map(lambda x: x.getter(),
                                                 compressed_model_loss_outputs_nested_obj_indexing))

        def match_fn(obj):
            for x in compressed_model_loss_outputs_nested_obj_indexing:
                if x.path == obj.path:
                    return True
            return False

        orig_model_loss_outputs = list(map(lambda x: x.getter(), filter(
            match_fn, orig_model_outputs_nested_obj_indexing.get_flat_nested_obj_indexing())))
        if len(orig_model_loss_outputs) == 0 or len(compressed_model_loss_outputs) == 0:
            return torch.zeros([], device=device)
        return reduce(
            lambda kd_loss, loss_tensors: kd_loss + kd_loss_fn(loss_tensors[0], loss_tensors[1]),
            zip(orig_model_loss_outputs, compressed_model_loss_outputs), torch.zeros([], device=device))

    @staticmethod
    def _is_loss(obj):
        if not isinstance(obj, torch.Tensor):
            return False
        if obj.requires_grad:
            return True
        return False

    def forward(self) -> torch.Tensor:
        """
        Gets knowledge distillation loss values from KnowledgeDistillationLossHandler, averages them in case of
        DataParallel execution (loss values for mini-batches) and frees up KnowledgeDistillationLossHandler loss values
        storage space.

        :return: Differentiable knowledge distillation loss value
        """
        loss = self._kd_loss_handler.get_kd_loss()
        if len(loss) == 0:
            raise RuntimeError('Empty list of loss tensors for KDLoss. Most likely compression_ctrl.loss()'
                               ' was called while model was in eval mode')
        for idx, _ in enumerate(loss):
            loss[idx] = loss[idx].unsqueeze(0)
        output = torch.cat(loss).mean()
        self._kd_loss_handler.zero_kd_loss()
        return output

    def statistics(self, quickly_collected_only=False):
        return {}
