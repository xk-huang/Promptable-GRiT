from typing import Dict, List, Optional, Tuple
import torch
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class GRiT(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        assert self.proposal_generator is not None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        # NOTE: For multiple run inference on the same image without re-encoding
        encoded_image_dict: Optional[dict] = None,
        replace_pred_boxes_with_gt_proposals: bool = False,
    ):
        assert not self.training
        assert detected_instances is None

        # NOTE: Modified to support multiple run inference on the same image without re-encoding
        if encoded_image_dict is None:
            encoded_image_dict = self.encode_image(batched_inputs)
        images = encoded_image_dict['images']
        features = encoded_image_dict['features']
        image_sizes = encoded_image_dict['image_sizes']

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        
        # NOTE: Modified to support multiple run inference on the same image without re-encoding
        results, _ = self.roi_heads(features, proposals, replace_pred_boxes_with_gt_proposals=replace_pred_boxes_with_gt_proposals)

        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return GRiT._postprocess(
                results, batched_inputs, image_sizes)
        else:
            return results

    def encode_image(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        image_sizes = images.image_sizes
        return dict(images=images, features=features, image_sizes=image_sizes)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        targets_task = batched_inputs[0]['task']
        for anno_per_image in batched_inputs:
            assert targets_task == anno_per_image['task']

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        proposals, roihead_textdecoder_losses = self.roi_heads(
            features, proposals, gt_instances, targets_task=targets_task)

        losses = {}
        losses.update(roihead_textdecoder_losses)
        losses.update(proposal_losses)

        return losses