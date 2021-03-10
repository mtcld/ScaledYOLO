# yolov4 scaled setup

### 1. python coco-to-yolo.py
```
a) This will convert the coco format dataset to yolo format.
```

### 2. python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 4 --img 1536 1536 --data coco.yaml --cfg yolov4-p7.yaml --weights yolov4-p7.pt --sync-bn --device 0,1 --name yolov4-p7
```
a) Download yolov4-p7.pt from https://github.com/gaurav67890/ScaledYOLO
b) coco.yaml contains train.txt, valid.txt and test.txt, each .txt file contains images for their respective classes.
```
### 3. python test.py --img 1536 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights runs/exp0_yolov4-p7/weights/last.pt
```
a) This will calculate the MAP for testing dataset--task test
```
