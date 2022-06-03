# Scaled YOLO-v4 repo setup

## clone repo
```
git clone https://github.com/thangnx183/ScaledYOLO.git
cd ScaledYOLO
git checkout update-docker
```

## setup docker 
- build docker container
```
nvidia-docker run --name ssd_scaled_yolo -it -v /mnt/ssd1/thang/ScaledYOLO:/workspace/share -v /mnt/ssd1/thang/coco_stuff_yolo:/workspace/share/coco_stuff_yolo -v /mnt/ssd1/thang/coco/datasets:/workspace/share/coco --shm-size=64g -p 6000:6000 -p 6001:6001 nvcr.io/nvidia/pytorch:21.08-py3

#/mnt/ssd1/thang/ScaledYOLO : path of ScaledYOLO folder
#/mnt/ssd1/thang/coco_stuff_yolo : path of data in yolo training format
#/mnt/ssd1/thang/coco/datasets : path of data in coco format

```
- minor change inside docker container due to bug occur when training
    - go inside torch pakage
    ```
    vim /opt/conda/lib/python3.8/site-packages/torch/_tensor.py

    ```
    - at line 647 change `return self.numpy()` to `return self.cpu().numpy()`

## install mish-cuda
```
# inside docker container
cd /workspace/share
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# go back to code folder
cd ../
```