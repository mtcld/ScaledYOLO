# yolov4 scaled setup

### 1. python convert-binary-mask.py
```
a) This will fetch the polygons from coco dataset and write binary mask, which will be used for training of unet model.
```

### 2. pytorch_resnet18_unet.ipynb
```
a) This notebook is used for training of scratch dataset using unet model, it will save the best model to best.pt

```

### 3. python inference.py
```
a) This will read the image and draw bounding boxes using yolov3 model and mask using unet model
```

