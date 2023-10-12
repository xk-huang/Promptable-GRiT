import inspect
import logging
import sys
import time
from argparse import Namespace

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


class PromptableGRiTGradioApp:
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

    DEFAULT_IMAGE = "https://farm9.staticflickr.com/8368/8446497097_62cb019310_z.jpg"
    # DEFAULT_IMAGE = "/home/v-yijicheng/xiaoke/GRiT/demo_images/000000438652.jpg"

    def build_app(self):
        with gr.Blocks() as app:
            with gr.Row():
                input_image = gr.Image(
                    value=self.DEFAULT_IMAGE,
                    label="Input Image",
                    interactive=True,
                    type="pil",
                    height=500,
                )
                output_image = gr.Image(label=f"Output Image", height=500)
            with gr.Row():
                input_image_botton = gr.Button(value="Encode image")
                input_image_text = gr.Textbox(
                    lines=1,
                    label="Input Image Text",
                    value="",
                    interactive=False,
                )

            with gr.Row():
                visual_prompt_mode = gr.Radio(
                    choices=["point", "box", "auto"],
                    value="auto",
                    label="Visual Prompt Mode",
                    interactive=True,
                )
                output_mode = gr.Radio(
                    choices=["image+text", "image", "text"],
                    value="image+text",
                    label="Output Mode",
                    interactive=True,
                )
            input_prompt_text = gr.Textbox(
                lines=1,
                label="Input Prompts: Points (x,y), or Box (x,y,x2,y2); Click image ot type in the text box.",
                value=self.DEFAULT_INPUT_PROMPT_TEXT,
                interactive=True,
            )
            prompt_input_botton = gr.Button(value="Run prediction head")
            output_text = gr.Textbox(
                lines=1, label="Output Text", value="", interactive=False
            )

            visual_prompt_mode.change(
                self._update_prompt_mode,
                inputs=[visual_prompt_mode],
                outputs=[input_prompt_text],
            )
            input_image.select(
                self._click_and_assign,
                inputs=[visual_prompt_mode, input_prompt_text],
                outputs=[input_prompt_text],
            )
            prompt_input_botton.click(
                self._run,
                inputs=[
                    input_image,
                    visual_prompt_mode,
                    input_prompt_text,
                    output_mode,
                ],
                outputs=[output_image, output_text],
            )
            input_image_botton.click(
                self._encode_image_with_botton,
                inputs=[input_image],
                outputs=[input_image_text],
            )

            # NOTE: call on app start
            # https://www.gradio.app/docs/blocks#blocks-load
            app.load(
                self._run,
                inputs=[
                    input_image,
                    visual_prompt_mode,
                    input_prompt_text,
                    output_mode,
                ],
                outputs=[output_image, output_text],
            )

        return app

    DEFAULT_INPUT_PROMPT_TEXT = "Please click images to assign prompts (point/box)."

    def _update_prompt_mode(self, visual_prompt_mode):
        if visual_prompt_mode == "point":
            logger.warning(f"point mode is box mode with tiny boxes.")
            return self.DEFAULT_INPUT_PROMPT_TEXT
        elif visual_prompt_mode == "box":
            return self.DEFAULT_INPUT_PROMPT_TEXT
        elif visual_prompt_mode == "auto":
            return ""
        else:
            raise ValueError(f"Unknown visual_prompt_mode: {visual_prompt_mode}")

    def _click_and_assign(
        self, visual_prompt_mode, input_prompt_text, evt: gr.SelectData
    ):
        x, y = evt.index
        if visual_prompt_mode == "point":
            input_prompt_text = f"{x},{y}"
        elif visual_prompt_mode == "box":
            if len(input_prompt_text.split(",")) == 2:
                input_prompt_text = f"{input_prompt_text},{x},{y}"
            else:
                input_prompt_text = f"{x},{y}"
        elif visual_prompt_mode == "auto":
            input_prompt_text = ""
        else:
            raise ValueError(f"Unknown visual_prompt_mode: {visual_prompt_mode}")
        return input_prompt_text

    def _run(
        self,
        input_image: Image.Image,
        visual_prompt_mode,
        input_prompt_text,
        output_mode,
    ):
        kwargs_dict = self._prompt_text2kwargs_dict(
            visual_prompt_mode, input_prompt_text
        )
        predictions = self._inference_model(
            input_image, visual_prompt_mode, **kwargs_dict
        )
        tic = time.time()

        if output_mode == "text":
            output_image = None
        else:
            output_image = self._post_process_output(input_image, predictions)
            logger.info(f"Post process time for visualization: {time.time() - tic}")
        if output_mode == "image":
            output_text = ""
        else:
            output_text = "\n".join(
                predictions["instances"].pred_object_descriptions.data
            )
        return output_image, output_text

    # NOTE: otherwise, OOM error
    @torch.no_grad()
    def _inference_model(
        self,
        input_image: Image.Image,
        visual_prompt_mode,
        input_boxes=None,
        input_points=None,
    ):
        if self.encoded_image_dict is None:
            self._encode_image_with_botton(input_image)
        aug_transform = self.aug_transform
        image = self.image
        inputs = self.inputs
        encoded_image_dict = self.encoded_image_dict

        if visual_prompt_mode == "auto":
            # NOTE: from detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN, use proposal_generator as usual
            self.model.proposal_generator = self.proposal_generator
            predictions = self.model.inference(
                [inputs],
                encoded_image_dict=encoded_image_dict,
                replace_pred_boxes_with_gt_proposals=False,
            )[
                0
            ]  # Automatically generate proposals
        else:
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
                replace_pred_boxes_with_gt_proposals=True,
            )[
                0
            ]  # Assign proposals
        return predictions

    def _encode_image_with_botton(self, input_image):
        aug_transform, image, inputs, encoded_image_dict = self._encode_image(
            input_image
        )
        self.aug_transform = aug_transform
        self.image = image
        self.inputs = inputs
        self.encoded_image_dict = encoded_image_dict

        return_string = [
            f"Image size: {input_image.size}",
            f"Aug image size: {image.shape}",
            f"aug_transform: {aug_transform}",
        ]
        return_string.append(
            "Features: "
            + ";".join(
                [f"{k}:{v.shape}" for k, v in encoded_image_dict["features"].items()]
            )
        )
        return "\n".join(return_string)

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

    @staticmethod
    def _prompt_text2kwargs_dict(visual_prompt_mode, prompt_text):
        kwargs_dict = {}
        if visual_prompt_mode == "point":
            x, y = list(map(int, prompt_text.replace(" ", "").split(",")))
            input_points = [[x, y]]
            kwargs_dict.update(dict(input_points=input_points))
        elif visual_prompt_mode == "box":
            x, y, x2, y2 = list(map(int, prompt_text.replace(" ", "").split(",")))
            input_boxes = [[x, y, x2, y2]]
            kwargs_dict.update(dict(input_boxes=input_boxes))
        elif visual_prompt_mode == "auto":
            pass
        else:
            raise ValueError(f"Unknown visual_prompt_mode: {visual_prompt_mode}")
        return kwargs_dict


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

promptable_det_gradio_app = PromptableGRiTGradioApp(args)
app = promptable_det_gradio_app.build_app()
app.launch()
