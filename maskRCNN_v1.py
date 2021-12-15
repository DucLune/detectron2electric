# %%
#!pip install pyyaml==5.1
#!pip install labelme
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html

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
import glob
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import pascal_voc,register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset,PascalVOCDetectionEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

# %%
#准备数据集合训练代码
#!rm -r detectron2electric/
#!git clone https://github.com/DucLune/detectron2electric.git
#!cp /content/detectron2electric/traincode/* ./ -rf

# %%
# %%
# Some basic setup:
# Setup detectron2 logger
setup_logger()

# 创建解析
parser = argparse.ArgumentParser(description="detectron2 demo",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 添加参数
parser.add_argument('--train_url', type=str, default="model/",help='the path model saved')
parser.add_argument('--data_url', type=str,default="detectron2electric/dataset/", help='the training data')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--dataset', type=str, default='coco',help='the dataset dirname')
parser.add_argument('--num_classes',type=int,default=20,help='cfg.MODEL.ROI_HEADS.NUM_CLASSES ')
# 解析参数
args, unkown = parser.parse_known_args()



# %%
from predictor import VisualizationDemo
#from detectron2electric.util import pictureUtils
#from detectron2electric.util import labelme2coco

# %%
#!rm -r detectron2electric/dataset/coco
#pictureUtils.buildCocoDataset("detectron2electric/SmallJPGImages","detectron2electric/Annotations/glue_pin_inclined_side_pin_glue_injection_hole_regular_part",os.path.join(args.data_url,args.dataset),0.7,0.3,0)
#labelme_json = glob.glob(os.path.join(args.data_url,args.dataset,"train")+'/*.json')
#labelme2coco.labelme2coco(labelme_json, os.path.join(args.data_url,args.dataset,"annotations","instances_train.json"))
#labelme_json = glob.glob(os.path.join(args.data_url,args.dataset,"val")+'/*.json')
#labelme2coco.labelme2coco(labelme_json, os.path.join(args.data_url,args.dataset,"annotations","instances_val.json"))

# %%
# %%
# 定义训练集和测试集
register_coco_instances("mydataset_train", {}, os.path.join(args.data_url, args.dataset,"annotations","instances_train2014.json"),os.path.join(args.data_url,"SmallJPGImages"))
register_coco_instances("mydataset_val", {}, os.path.join(args.data_url, args.dataset,"annotations","instances_val2014.json"), os.path.join(args.data_url,"SmallJPGImages"))

# %%
# %%
# 定义模型并训练
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(args.data_url,"model_final_f10217.pkl")
cfg.MODEL.DEVICE = args.device
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.DATASETS.TRAIN = ("mydataset_train",)
cfg.DATASETS.TEST = ("mydataset_val",)


cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.STEPS = []        # do not decay learning rate
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = args.train_url
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
if args.device != 'cpu':
    trainer.train()

# %%
# %%
# 验证集验证

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
# 加载训练出来的权重
cfg.MODEL.WEIGHTS = args.train_url+"/model_final.pth"
# 构建评估器
evaluator = COCOEvaluator("mydataset_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "mydataset_val")
predictor = DefaultPredictor(cfg)
# 输出模型在验证集上的性能指标
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
#model = trainer.build_model(cfg)
#metrics = trainer.test(cfg, model, evaluator)

# %%
#!rm model/*.jpg
#!mkdir model

# 在测试集上测试，并保存图片
demo = VisualizationDemo(cfg)
#!rm model/*.jpg
#!mkdir model
import os
filePath = os.path.join(args.data_url,args.dataset,"test2014")
list_data=os.listdir(filePath)
for filename in list_data:
  if filename.split('.')[1]=='json':
    continue
  im = cv2.imread(os.path.join(filePath, filename))
  predictions, visualized_output = demo.run_on_image(im)
  visualized_output.save(os.path.join(args.train_url, filename))



