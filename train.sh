python -m torch.distributed.launch --nproc_per_node 3 train.py --batch-size 9 --img 1024 1024 --data data/coco.yaml --cfg models/yolov4-p7.yaml --weights 'checkpoints/26_8/dent/pretrain.pt' --sync-bn --device 0,1,2 --name dent_merimen_19_02










#python -m torch.distributed.launch --nproc_per_node 3 train.py --batch-size 9 --img 1024 1024 --hyp data/hyp.pretrain.yaml --data data/coco.yaml --cfg models/yolov4-p7.yaml --weights 'checkpoints/26_8/dent/pretrain.pt' --sync-bn --device 0,1,2 --name dent_merimen_19_02


#nvidia-docker run --name yolor -it -v /mnt/ssd1/thang/yolor:/workspace/share -v /mnt/ssd1/thang/coco_stuff_yolo:/workspace/share/coco_stuff_yolo -v /data1/thang/datasets:/workspace/share/coco --shm-size=64g -p 6000:6000 -p 6001:6001 nvcr.io/nvidia/pytorch:21.08-py3