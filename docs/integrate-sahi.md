# integrate SAHI into Scale-yolo-v4

## install SAHI lib
- download lib inside Scale-yolo-v4 folder
```
cd ScaledYOLO
git clone https://github.com/thangnx183/sahi.git
cd sahi
git checkout integrate-scale-yolo-v4
python -m pip install -e .
```

## checkout integrate-SAHI branch of Scale-yolo-v4 repo
```
git checkout integrate-SAHI
```

## Demo and detail of how to use sahi with scale-yolo-v4 in [demo_yolo_v4.py](demo_yolo_v4.py) , general guide of sahi [here](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb)