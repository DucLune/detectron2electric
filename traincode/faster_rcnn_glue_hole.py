# %%
# import some common libraries
import random
import cv2
import json
import os
import numpy as np
import argparse
import torch
import detectron2
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import pascal_voc
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset,PascalVOCDetectionEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

import myPascal

# %%
# Some basic setup:
# Setup detectron2 logger
setup_logger()

# 创建解析
parser = argparse.ArgumentParser(description="detectron2 demo",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 添加参数
parser.add_argument('--train_url', type=str, default=os.getcwd()+"/model/glue_hole/",
                    help='the path model saved')
parser.add_argument('--data_url', type=str,
                    default=os.getcwd()+"/dataset/", help='the training data')
parser.add_argument('--device', type=str, default='cpu',
                    help='the training device')
parser.add_argument('--dataset', type=str, default='V003',
                    help='the dataset dirname')
# 解析参数
args, unkown = parser.parse_known_args()


# %%
# 定义训练集和测试集
class_names = ("glue","injection_hole","pin_glue","pin_inclined_side","pin_inclined_top")
#class_names = ("Component", "Injection_hole")
dirname = os.path.join(args.data_url, args.dataset)
myPascal.register_pascal_voc(
    "mydataset_train", dirname, split="train", year=2021, class_names=class_names)
myPascal.register_pascal_voc(
    "mydataset_test", dirname, split="test", year=2021, class_names=class_names)


# %%
# 定义模型并训练
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file(
    "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = args.data_url+"/model_final_b1acc2.pkl"
cfg.MODEL.DEVICE = args.device
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.DATASETS.TRAIN = ("mydataset_train",)
cfg.DATASETS.TEST = ("mydataset_test",)


cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []        # do not decay learning rate
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = args.train_url
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
if args.device != 'cpu':
    trainer.train()


# %%
# 定义模型并训练

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = args.train_url+"/model_final.pth"

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
evaluator = COCOEvaluator("mydataset_test", output_dir=cfg.OUTPUT_DIR)

val_loader = build_detection_test_loader(cfg, "mydataset_test")
#print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
model = trainer.build_model(cfg)
metrics = trainer.test(cfg, model, evaluator)
predictor = DefaultPredictor(cfg)
demo = VisualizationDemo(cfg)
# 从文件中获取要拷贝的文件的信息


def get_filename_from_txt(file):
    filename_lists = []
    with open(file, 'r', encoding='utf-8') as f:
        lists = f.readlines()
        for list in lists:
            filename_lists.append(str(list).strip('\n')+'.jpg')
    return filename_lists


filename_lists = get_filename_from_txt(
    os.path.join(dirname, "ImageSets", "Main", "test.txt"))
for filename in filename_lists:
    im = cv2.imread(os.path.join(dirname, "JPEGImages", filename))
    predictions, visualized_output = demo.run_on_image(im)
    visualized_output.save(os.path.join(args.train_url, filename))


# %%




