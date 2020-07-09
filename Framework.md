# Framework

## Environment setup

!pip install -U torch==1.5 torchvision == 0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html   
!pip install pyyaml == 5.1 pycocotools>=2.0.1  
import torch, torchvision  
print(torch.__version__, torch.cuda.is_available())  
!gcc --version  
!pip install detectron2 == 0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html  
!pip install fvcore  
!pip uninstall -y opencv-python opencv-contrib-python  
!apt install python3-opencv  

## Build Dataset

There are two selection:
1. transform the dataset into COCO format and regist it by 

from detectron2.data.datasets import register_coco_instances  
register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")  
register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")  

2. build a fuction to get image and annotataion imformation loaded as COCO format  


Using DatasetCatalog, MetadataCatalog to store the imformation of dataset and metadata  

## Build custom dataloader to implement data augmentation 

build a custom mapper and use build_detection_train_loader(cfg, mapper=train_mapper) to replace trainer.build_train_loader

## Implement custom backbone

reference: https://github.com/youngwanLEE/vovnet-detectron2

build a backbone and regist-> set parameter in config, merge from file call the config file and build the whole model.


## Inference and comparison



| Model | AP | Inference Time | Memory| 
| -------- | -------- | -------- |---|
|Mask-RCNN R50-FPN|96.41|0.5929|2797M|
|Mask-RCNN R101-FPN|97.11|0.7518|4212M|
|Mask-RCNN X101-FPN|97.94|0.7784|6088M|


## Bonus

1. as custom backbone showed
2. pros:容易搭建、內建模型多 cons: 層層套件包裝不易調整、有些小bug、內建backbone選項較少
3. pass
4. 可搭配ocr model 進行身分資料讀取可用於自動化建檔、機場快速通關、各式門禁系統等等