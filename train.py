import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from Aug import mapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.transforms import RandomContrast,RandomRotation,RandomBrightness
from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import DatasetFromList, MapDataset
import logging
import copy
from typing import List, Optional, Union
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

register_coco_instances("my_dataset", {}, "/content/gdrive/My Drive/detectron2/data/midv500_coco/midv500_coco.json", "/content/gdrive/My Drive/detectron2/data/midv500/")
register_coco_instances("test_dataset", {}, "/content/gdrive/My Drive/detectron2/data/midv500_coco/midv2019_coco.json", "/content/gdrive/My Drive/detectron2/data/midv-2019/")

dataset_dicts = DatasetCatalog.get("my_dataset")
metadata = MetadataCatalog.get("my_dataset")
testdata_dicts = DatasetCatalog.get("test_dataset")
test_meta = MetadataCatalog.get("test_dataset")


tfm = [RandomContrast(intensity_min=0.8, intensity_max=2), 
       RandomRotation((-30,30), expand=True, center=None, sample_style='range', interp=None),
      RandomBrightness(0.8, 1.2)]



cfg = get_cfg()
train_mapper = mapper(cfg, augmentations= tfm , image_format= 'BGR')



cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.DATASETS.TRAIN = ('my_dataset',)
cfg.DATASETS.TEST = ()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.build_train_loader = build_detection_train_loader(cfg, mapper=train_mapper)
trainer.resume_or_load(resume=False)
trainer.train()

from contextlib import redirect_stdout

#ouput config
with open('test.yaml', 'w') as f:
    with redirect_stdout(f): print(cfg.dump())

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "test_dataset")
inference_on_dataset(trainer.model, val_loader, evaluator)
