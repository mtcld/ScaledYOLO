# Scaled YOLO-v4 repo setup

## clone repo
```
git clone https://github.com/thangnx183/ScaledYOLO.git
cd ScaledYOLO
git checkout develop
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

# How to run
## Convert coco format data into coco_stuff format (yolo training format)
- COCO format data plit into 2 parts : old dataset and merimen dataset
```
/workspace/share/coco/
└───origin_crack/                            #old dataset
│   └───annotations/
│   │       │   train.json
│   │       │   test.json
│   │       │   valid.json
│   │   
│   └───images/
│
└───merimen_coco/06_04_2022/origin_crack/   #merimen dataset
    └───annotations/
    │       │   merimen.json
    │   
    └───images/
```
### Convert datasets using `coco-to-yolo.py` file 
- change variable with corresponding value : 
    - `mode` : list name of files in annotation 
    - `damage` : damage name
    - `annt_dir` : path to annotations folder  
    - `img_dir` : path to images folder
- example : 
    - convert old dataset
    ```
    # modify coco-to-yolo.py file
    mode = ['train','valid','test']
    damage = 'origin_crack'
    annt_dir = 'coco/'+damage+'/annotations'
    img_dir = 'coco/'+damage+'/images'

    #run script : python coco-to-yolo.py
    ```
    - convert merimen dataset 
    ```
    # modify coco-to-yolo.py file
    mode = ['merimen']
    damage = 'origin_crack'
    annt_dir = 'coco/merimen_coco/06_04_2022/'+damage+'/annotations'
    img_dir = 'coco/merimen_coco/06_04_2022/'+damage+'/images'

    #run script : python coco-to-yolo.py
    ```
    - After convert , inside `coco_stuff_yolo` folder 
    ```
    /workspace/share/coco_stuff_yolo/
    └───origin_crack/                            
        └───train/             # each files in this folder is annotation for corresponing image
        └───train.txt          # list of path to each image in train
        └───valid/
        └───valid.txt
        └───test/
        └───test.txt
        └───merimen/
        └───merimen.txt
    ``` 
### Merge merimen data into train using `concat_2_train_files.py`
```
python concat_2_train_files.py coco_stuff_yolo/origin_crack/train.txt coco_stuff_yolo/origin_crack/merimen.txt  coco_stuff_yolo/origin_crack/merge_merimen_train.txt
```

## How to train
- modify `data/coco.yaml` with path of train, valid, test files
    - example : 
        ```
        # train and val datasets (image directory or *.txt file with image paths)
        train: coco_stuff_yolo/origin_crack/merge_merimen_train.txt  # 118k images
        val: coco_stuff_yolo/origin_crack/valid.txt  # 5k images
        test: coco_stuff_yolo/origin_crack/test.txt  # 

        # number of classes
        nc: 1

        # class names
        names : ['crack']
        ```
- download [YOLO-v4 P7](https://drive.google.com/file/d/18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3/view) model
- change arguments of training script `train.sh` following `train.py` file
    - example `train.sh`
    ```
    python -m torch.distributed.launch --nproc_per_node 3 train.py --batch-size 9 --img 1024 1024 --hyp data/hyp.finetune.yaml --data data/coco.yaml --cfg models/yolov4-p7.yaml --weights 'yolov4-p7.pt' --sync-bn --device 0,1,2 --name origin_crack_merimen

    ```
- run training script 
    ```
    ./train.sh
    ```