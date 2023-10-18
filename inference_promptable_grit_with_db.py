import inspect
import json
import logging
import sys
import time
from argparse import Namespace
import multiprocessing as mp
import os
import sqlite3
from contextlib import closing

import detectron2.data.transforms as T
import numpy as np
import torch
import tqdm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import ColorMode
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import datasets

sys.path.insert(0, "third_party/CenterNet2/projects/CenterNet2/")
from centernet.config import add_centernet_config
from grit.config import add_grit_config
from grit.predictor import Visualizer_GRiT

logger = logging.getLogger(__name__)
filname = os.getenv("LOG_FILENAME", "inference_promptable_grit.log")
logging.basicConfig(level=logging.INFO, filename=filname, filemode="w", force=True)


def prepare_dataset(split="test"):
    # NOTE: the data script is customized for my another project
    # in case you want to inference on your own dataset, you need to write your own data script
    path = "third_party/data_scripts/visual_genome-densecap-local.py"
    name = "densecap"
    if os.getenv("USE_GRIT_SPLIT") is not None:
        path = "third_party/data_scripts/visual_genome-grit-local.py"
        name = "grit"
    # split = None
    cache_dir = ".data.cache"
    streaming = False
    with_image = True
    # NOTE: change this accordingly
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


class InferenceDataset(Dataset):
    def __init__(self, dataset, sample_ids=None):
        self.dataset = dataset
        if sample_ids is None:
            self.num_samples = len(dataset)
        else:
            self.num_samples = len(sample_ids)
            self.dataset = [self.dataset[i] for i in sample_ids]
            logger.info(f"len(self.dataset): {len(self.dataset)}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.convert_sample_format(self.dataset[idx], idx)

    def convert_sample_format(self, sample, idx):
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
        image_id = sample["image_id"]

        regions = sample["regions"]
        input_boxes = []
        gt_captions = []
        region_ids = []
        for region in regions:
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            x2, y2 = x + w, y + h
            input_box = [x, y, x2, y2]
            input_boxes.append(input_box)
            gt_captions.append(region["phrases"])
            region_ids.append(region["region_id"])
        # import matplotlib.pyplot as plt
        # # Plot the image
        # input_image_np = np.array(input_image)
        # fig, ax = plt.subplots(1)
        # ax.imshow(input_image_np)
        # for box in input_boxes:
        #     x, y, x2, y2 = box
        #     w, h = x2 - x, y2 - y
        #     rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # input_boxes = np.array(input_boxes)
        # x_min, y_min, x_max, y_max = input_boxes[:, 0].min(), input_boxes[:, 1].min(), input_boxes[:, 2].max(), input_boxes[:, 3].max()
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        # plt.savefig("test_bbox.png")

        input_boxes, gt_captions, region_ids = self._normalize_regions(
            idx, input_image, image_id, input_boxes, gt_captions, region_ids
        )

        # # Plot the image
        # input_image_np = np.array(input_image)
        # fig, ax = plt.subplots(1)
        # ax.imshow(input_image_np)
        # for box in input_boxes:
        #     x, y, x2, y2 = box
        #     w, h = x2 - x, y2 - y
        #     rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # x_min, y_min, x_max, y_max = input_boxes[:, 0].min(), input_boxes[:, 1].min(), input_boxes[:, 2].max(), input_boxes[:, 3].max()
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        # plt.savefig("test_bbox.after_norm.png")
        # import IPython; IPython.embed()

        return dict(
            input_image=input_image,
            input_boxes=input_boxes,
            gt_captions=gt_captions,
            image_id=image_id,
            region_ids=region_ids,
        )

    def _normalize_regions(
        self, idx, input_image, image_id, input_boxes, gt_captions, region_ids
    ):
        image_width = input_image.width
        image_height = input_image.height
        input_boxes = np.asarray(input_boxes)

        if np.any(input_boxes < 0):
            corrupted_indices = np.where(input_boxes < 0)[0]
            corrupted_boxes = input_boxes[corrupted_indices]
            corrupted_region_ids = [region_ids[i] for i in corrupted_indices]
            logger.error(
                f"[{image_id}({idx})]\tinput_boxes < 0, got\t{corrupted_boxes} at {image_id}({idx})-{corrupted_region_ids}({corrupted_indices})"
            )
            input_boxes = np.clip(input_boxes, 0, None)

        if np.any(input_boxes[:, [0, 2]] >= image_width):
            corrupted_indices = np.where(input_boxes[:, [0, 2]] >= image_width)[0]
            corrupted_boxes = input_boxes[corrupted_indices]
            corrupted_region_ids = [region_ids[i] for i in corrupted_indices]
            logger.error(
                f"[{image_id}({idx})]\tinput_boxes[:, [0, 2]] >= image_width({image_width}),\tgot {corrupted_boxes} at {image_id}({idx})-{corrupted_region_ids}({corrupted_indices})"
            )
            input_boxes[:, [0, 2]] = np.clip(
                input_boxes[:, [0, 2]], None, image_width - 1
            )

        if np.any(input_boxes[:, [1, 3]] >= image_height):
            corrupted_indices = np.where(input_boxes[:, [1, 3]] >= image_height)[0]
            corrupted_boxes = input_boxes[corrupted_indices]
            corrupted_region_ids = [region_ids[i] for i in corrupted_indices]
            logger.error(
                f"[{image_id}({idx})]\tinput_boxes[:, [1, 3]] >= image_height({image_height}),\tgot {corrupted_boxes} at {image_id}({idx})-{corrupted_region_ids}({corrupted_indices})"
            )
            input_boxes[:, [1, 3]] = np.clip(
                input_boxes[:, [1, 3]], None, image_height - 1
            )

        nonempty_bool = self.nonempty(input_boxes)
        if np.any(~nonempty_bool):
            corrupted_indices = np.where(~nonempty_bool)[0]
            corrupted_boxes = input_boxes[corrupted_indices]
            corrupted_region_ids = [region_ids[i] for i in corrupted_indices]
            logger.error(
                f"[{image_id}({idx})]\tempty box,\tgot {corrupted_boxes} at {image_id}({idx})-{corrupted_region_ids}({corrupted_indices})"
            )
            # nonempty_indices = np.where(nonempty_bool)[0]
            # input_boxes = input_boxes[nonempty_indices]
            # gt_captions = [gt_captions[i] for i in nonempty_indices]
            # region_ids = [region_ids[i] for i in nonempty_indices]
            empty_indices = np.where(~nonempty_bool)[0]
            input_boxes[empty_indices] = np.array([0, 0, 1, 1])
            logger.warning(
                f"[{image_id}({idx})]\tset empty box to [0, 0, 1, 1]\tgot {corrupted_boxes} at {image_id}({idx})-{corrupted_region_ids}({corrupted_indices})"
            )

        return input_boxes, gt_captions, region_ids

    @staticmethod
    def nonempty(box, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def build_dataloader(self, batch_size=1, num_workers=4, shuffle=False):
        if batch_size != 1:
            raise ValueError("batch_size should be 1 in GRiT")

        def _collate_func(batch):
            return batch[0]

        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=_collate_func,
        )


# NOTE: Test the dataset loading
if os.getenv("TEST_DATASET_LOADING") is not None:
    split = os.getenv("TEST_DATASET_LOADING_SPLIT", "test")
    eval_dataset = prepare_dataset(split=split)
    eval_infer_dataset = InferenceDataset(eval_dataset)
    logger.info(f"TEST_DATASET_LOADING: {os.getenv('TEST_DATASET_LOADING')}")
    for data in tqdm.tqdm(eval_infer_dataset.build_dataloader(num_workers=10)):
        pass
    exit()

if os.getenv("RUN_CORRUPTED_BOXES") is not None:
    eval_dataset = prepare_dataset(split="test")
    eval_infer_dataset = InferenceDataset(eval_dataset)
    logger.info(f"RUN_CORRUPTED_BOXES: {os.getenv('RUN_CORRUPTED_BOXES')}")
    # NOTE: the real corrpted 1864, ValueError: len(pred_captions) != len(gt_captions): 62 != 63 at 4
    # corrupted_boxes_ids = [295,528,697,1369,2049,2292,2569,2779,3393,3482,3685,3810,4794,4811]
    corrupted_boxes_ids = [1864]
    eval_infer_dataset = InferenceDataset(eval_dataset, sample_ids=corrupted_boxes_ids)
    for data in tqdm.tqdm(eval_infer_dataset.build_dataloader(num_workers=0)):
        pass


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

    def inference_dataset(
        self,
        dataset: InferenceDataset,
        split_name="inference",
        max_samples=None,
        db_file="inference_results.db",
    ):
        """_summary_

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """

        def save_results(queue, db_file):
            with closing(sqlite3.connect(db_file)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(_id) FROM results")
                max_id = cursor.fetchone()[0]
                if max_id is None:
                    region_cnt = 0
                else:
                    region_cnt = max_id + 1

                while True:
                    batch = queue.get()
                    if batch is None:
                        break

                    pred_captions = batch["pred_captions"]
                    gt_captions = batch["gt_captions"]
                    input_boxes = batch["input_boxes"]
                    if isinstance(input_boxes, np.ndarray):
                        input_boxes = input_boxes.tolist()
                    region_ids = batch["region_ids"]

                    image_id = batch["image_id"]
                    split_name = batch["split_name"]

                    for pred_caption, gt_caption, input_box, region_id in zip(
                        pred_captions, gt_captions, input_boxes, region_ids
                    ):
                        result = (
                            region_cnt,
                            split_name,
                            json.dumps(gt_caption),
                            json.dumps(pred_caption),
                            json.dumps(input_box),
                            image_id,
                            region_id,
                            json.dumps([1.0]),
                        )
                        region_cnt += 1

                        conn.execute(
                            """  
                            INSERT INTO results (  
                                _id, split, _references, candidates,  
                                metadata_input_boxes, metadata_image_id, metadata_region_id,  
                                logits_iou_scores  
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)  
                        """,
                            result,
                        )

                    # NOTE: commit after each batch
                    conn.commit()

        def result_exists(db_file, image_id, region_id):
            with closing(sqlite3.connect(db_file)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """  
                    SELECT COUNT(*) FROM results  
                    WHERE metadata_image_id = ? AND metadata_region_id = ?  
                """,
                    (image_id, region_id),
                )
                count = cursor.fetchone()[0]
            return count > 0

        def init_database(db_file):
            with closing(sqlite3.connect(db_file)) as conn:
                with conn:
                    conn.execute(
                        """  
                        CREATE TABLE IF NOT EXISTS results (  
                            _id INTEGER PRIMARY KEY,  
                            split TEXT,  
                            _references TEXT,  
                            candidates TEXT,  
                            metadata_input_boxes TEXT,  
                            metadata_image_id INTEGER,  
                            metadata_region_id INTEGER,  
                            logits_iou_scores TEXT  
                        )  
                    """
                    )

        # NOTE: num_workers=0 is for debugging
        dataloader = dataset.build_dataloader()

        if max_samples is None:
            max_samples = len(dataset)

        # Initialize the SQLite database and start the save_results process
        init_database(db_file)

        # Create a queue to store the results and start the saving process
        result_queue = mp.Queue(maxsize=50)
        save_process = mp.Process(target=save_results, args=(result_queue, db_file))
        save_process.start()

        results = []
        for sample_id, batch in enumerate(tqdm.tqdm(dataloader)):
            if sample_id == max_samples:
                break

            input_image = batch["input_image"]
            input_boxes = batch["input_boxes"]
            gt_captions = batch["gt_captions"]
            image_id = batch["image_id"]
            region_ids = batch["region_ids"]

            if all(
                result_exists(db_file, image_id, region_id) for region_id in region_ids
            ):
                continue

            pred_captions = self.inference_text(input_image, input_boxes)
            if len(pred_captions) != len(gt_captions):
                raise ValueError(
                    f"len(pred_captions) != len(gt_captions): {len(pred_captions)} != {len(gt_captions)} at {sample_id}"
                )
            result_queue.put(
                dict(
                    pred_captions=pred_captions,
                    gt_captions=gt_captions,
                    input_boxes=input_boxes,
                    image_id=image_id,
                    region_ids=region_ids,
                    split_name=split_name,
                )
            )

        # Signal the saving process to finish
        result_queue.put(None)

        # Wait for the saving process to complete and get the results
        save_process.join()

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
        aug_transform, image, inputs, encoded_image_dict = self._encode_image(
            input_image
        )

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
args.config_file = "configs/GRiT_B_DenseCap.yaml"
# args.config_file = "configs/GRiT_B_DenseCap_ObjectDet.yaml"
args.opts = [
    "MODEL.WEIGHTS",
    "models/grit_b_densecap.pth",
    # "models/grit_b_densecap_objectdet.pth",
]
args.confidence_threshold = 0.5
args.test_task = "DenseCap"
args.cpu = False


infer_engine = PromptableGRiTInferenceEngine(args)

if os.getenv("RUN_CORRUPTED_BOXES") is None:
    eval_dataset = prepare_dataset(split="test")
    eval_infer_dataset = InferenceDataset(eval_dataset)

# NOTE: max_samples=10 is for debugging
max_samples = os.getenv("MAX_SAMPLES", None)
if max_samples is not None:
    max_samples = int(max_samples)
db_file = "inference_results.db"
results = infer_engine.inference_dataset(
    eval_infer_dataset,
    split_name="inference",
    max_samples=max_samples,
    db_file=db_file,
)


def convert_db_to_json(db_file, json_file):
    with closing(sqlite3.connect(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """  
            SELECT _id, split, _references, candidates, metadata_input_boxes, metadata_image_id, metadata_region_id, logits_iou_scores
            FROM results  
        """
        )
        results = cursor.fetchall()
    results = [
        dict(
            _id=_id,
            split=split,
            references=json.loads(gt_captions),
            candidates=json.loads(pred_captions),
            metadata=dict(
                metadata_input_boxes=json.loads(input_boxes),
                metadata_image_id=image_id,
                metadata_region_id=region_id,
            ),
            logits=dict(iou_scores=json.loads(iou_scores)),
        )
        for _id, split, gt_captions, pred_captions, input_boxes, image_id, region_id, iou_scores in results
    ]
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)


json_file = "grit-vg-densecap-local.json"
convert_db_to_json(db_file, json_file)
