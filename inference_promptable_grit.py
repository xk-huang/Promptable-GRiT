import datasets
import inspect
import logging
import sys
import time
from argparse import Namespace
import tqdm

import detectron2.data.transforms as T
import gradio as gr
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import ColorMode
from PIL import Image
import numpy as np

sys.path.insert(0, "third_party/CenterNet2/projects/CenterNet2/")
from centernet.config import add_centernet_config
from grit.config import add_grit_config
from grit.predictor import Visualizer_GRiT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)


def prepare_dataset(split="test"):
    # NOTE: the data script is customized for my another project
    # in case you want to inference on your own dataset, you need to write your own data script
    path = "/home/v-yijicheng/xiaoke/segment-caption-anything-v2/src/data/data_scripts/visual_genome-densecap-local.py"
    name = "densecap"
    # split = None
    cache_dir = "/home/v-yijicheng/xiaoke/segment-caption-anything-v2/.data.cache"
    streaming = False
    with_image = True
    base_dir = "/mnt/onemodel/data_raw/VisualGenome/"
    base_annotation_dir = "/mnt/onemodel/data_raw/VisualGenome/annotations/"

    dataset = datasets.load_dataset(
        path=path,
        name=name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        with_image=with_image,
        base_dir=base_dir,
        base_annotation_dir=base_annotation_dir,
    )

    return dataset


class PromptableGRiTInferenceEngine:
    def setup_cfg(self, args):
        cfg = get_cfg()
        if args.cpu:
            cfg.MODEL.DEVICE = "cpu"
        add_centernet_config(cfg)
        add_grit_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            args.confidence_threshold
        )
        if args.test_task:
            cfg.MODEL.TEST_TASK = args.test_task
        cfg.MODEL.BEAM_SIZE = 1
        cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
        cfg.USE_ACT_CHECKPOINT = False
        cfg.freeze()
        return cfg

    def __init__(self, args):
        # TODO: move obj outside of the class, for gradio reload
        # The size of the image may be changed by the augmentations.
        # NOTE: from demo/demo.py
        cfg = self.setup_cfg(args)
        # self.predictor = DefaultPredictor(cfg)

        # NOTE: from demo/predictor.py:VisualizationDemo
        cpu_device = torch.device("cpu")
        instance_mode = ColorMode.IMAGE

        # NOTE: from detectron2/engine/defaults.py:DefaultPredictor
        model = build_model(cfg)
        model.eval()
        logger.info(f"meta_arch: {type(model)} from {inspect.getfile(type(model))}")

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        input_format = cfg.INPUT.FORMAT
        assert input_format in ["RGB", "BGR"], input_format

        # NOTE: from detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN, only by setting `proposal_generator=None`, the model will use the proposals in inputs
        # Otherwise it will generate proposals by itself
        # NOTE: arg `detected_instances` is the detection results, which is for mask and keypoint (at least in GeneralizedRCNN). It is not the proposals.
        self.proposal_generator = model.proposal_generator

        self.cfg = cfg
        self.cpu_device = cpu_device
        self.instance_mode = instance_mode
        self.model = model
        self.aug = aug
        self.input_format = input_format

        self.aug_transform = None
        self.image = None
        self.inputs = None
        self.encoded_image_dict = None

    def _encode_image_with_botton(self, input_image):
        aug_transform, image, inputs, encoded_image_dict = self._encode_image(
            input_image
        )
        self.aug_transform = aug_transform
        self.image = image
        self.inputs = inputs
        self.encoded_image_dict = encoded_image_dict

    @torch.no_grad()
    def _encode_image(self, input_image):
        if not isinstance(input_image, Image.Image):
            raise ValueError(
                f"input_image should be PIL.Image.Image, got {type(input_image)}"
            )
        original_image = convert_PIL_to_numpy(input_image, self.input_format)
        # return self.predictor(original_image)

        height, width = original_image.shape[:2]
        aug_transform = self.aug.get_transform(original_image)
        image = aug_transform.apply_image(original_image)
        image = torch.as_tensor(
            image.astype("float32").transpose(2, 0, 1)
        )  # HWC -> CHW
        logger.info(
            f"original_image.shape: {original_image.shape}; image.shape: {image.shape}"
        )

        inputs = {"image": image, "height": height, "width": width}
        encoded_image_dict = self.model.encode_image([inputs])
        return aug_transform, image, inputs, encoded_image_dict

    def inference_text(self, input_image, input_boxes):
        """_summary_

        Args:
            input_image (_type_): PIL image
            input_boxes (_type_): Nx4 np array or list of list. The coordinates are in the original image space.

        Returns:
            _type_: return list of list of str
        """
        predictions = self._inference_model(input_image, input_boxes)
        pred_captions = predictions["instances"].pred_object_descriptions.data

        if isinstance(pred_captions[0], str):
            pred_captions = [[i] for i in pred_captions]
        return pred_captions

    def inference_visualization(self, input_image, input_boxes):
        """_summary_

        Args:
            input_image (_type_): PIL image
            input_boxes (_type_): Nx4 np array or list of list. The coordinates are in the original image space.

        Returns:
            _type_: np.array, The visualization of the inference result. By GRiT adapted Detectron2 visualizer
        """
        predictions = self._inference_model(input_image, input_boxes)
        output_image = self._post_process_output(input_image, predictions)

        return output_image

    # NOTE: GRiT only supports batch size 1
    # NOTE: otherwise, OOM error
    @torch.no_grad()
    def _inference_model(
        self,
        input_image: Image.Image,
        input_boxes=None,
        input_points=None,
    ):
        if self.encoded_image_dict is None:
            self._encode_image_with_botton(input_image)
        aug_transform = self.aug_transform
        image = self.image
        inputs = self.inputs
        encoded_image_dict = self.encoded_image_dict

        if input_points is None and input_boxes is None:
            raise ValueError("input_points or input_boxes should not be None.")
        elif input_points is not None and input_boxes is not None:
            raise ValueError(
                "input_points and input_boxes should not be both not None."
            )
        elif input_points is not None and input_boxes is None:
            input_points = np.asarray(input_points)
            input_boxes = np.concatenate([input_points, input_points], axis=-1)

        # NOTE: from detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN, only by setting `proposal_generator=None`, the model will use the proposals in inputs
        self.model.proposal_generator = None
        # NOTE: from fvcore/transforms/transform.py:apply_box, which takes in Nx4 np array.
        # It call `detectron2/data/transforms/transform.py:apply_coords` which casts the dtype to fp32.
        input_boxes = aug_transform.apply_box(input_boxes)
        input_height, input_width = image.shape[-2:]
        num_boxes = len(input_boxes)
        proposals = Instances(
            (input_height, input_width),
            proposal_boxes=Boxes(input_boxes),
            scores=torch.ones(num_boxes),
            objectness_logits=torch.ones(num_boxes),
        )
        inputs.update(dict(proposals=proposals))

        # NOTE: The **batch system** of Detectron2 is to **use List**
        predictions = self.model.inference(
            [inputs],
            encoded_image_dict=encoded_image_dict,
            replace_pred_boxes_with_proposals=True,
        )[
            0
        ]  # Assign proposals

        return predictions

    def _post_process_output(self, image, predictions):
        visualizer = Visualizer_GRiT(image, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return vis_output.get_image()


args = Namespace()
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
args.config_file = "configs/GRiT_B_DenseCap_ObjectDet.yaml"
args.opts = [
    "MODEL.WEIGHTS",
    "models/grit_b_densecap_objectdet.pth",
]
args.confidence_threshold = 0.5
args.test_task = "DenseCap"
args.cpu = False


infer_engine = PromptableGRiTInferenceEngine(args)

eval_dataset = prepare_dataset(split="test")


def convert_sample_format(sample):
    """_summary_

    Args:
        sample (_type_): _description_

    Returns:
        dict:
            input_image: PIL,
            input_boxes: list or list of int, Nx4
            gt_captions: list of list of str, Nx1

    """
    input_image = sample["image"]
    regions = sample["regions"]
    input_boxes = []
    gt_captions = []
    for region in regions:
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        x2, y2 = x + w, y + h
        input_box = [x, y, x2, y2]
        input_boxes.append(input_box)
        gt_captions.append(region["phrases"])

    return dict(
        input_image=input_image, input_boxes=input_boxes, gt_captions=gt_captions
    )


sample = eval_dataset[0]
input_dict = convert_sample_format(sample)
gt_captions = input_dict.pop("gt_captions")
infer_engine.inference_text(**input_dict)

import IPython

IPython.embed()
