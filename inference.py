import argparse
import cv2
import torch
from numpy import random
import time
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImagesBatch
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from collections import OrderedDict

class Dev_model():
    def __init__(self,weight_path,conf_score,iou_thres,imgsz = 1024):
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        self.conf_score = conf_score
        self.iou_thres = iou_thres

        self.model = attempt_load(weight_path, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size

        if self.half : 
            self.model.half()   
    
    def inference(self,img_paths):
        # create dataset
        dataset = LoadImagesBatch(img_paths,img_size=self.imgsz)

        # load dataset and create batch with size equal number of given img_paths
        batch = []
        shapes = []
        paths = []
        for path, img, im0s, shape in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            batch.append(img)
            shapes.append(shape)
            paths.append(path)
        batch = torch.cat(batch)

        # run batch inference
        with torch.no_grad():
            pred = self.model(batch, augment=False)[0]

        # Apply MNS
        pred = non_max_suppression(pred, self.conf_score, self.iou_thres, agnostic=True)

        # Post process
        output = OrderedDict()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # remove batch padding and rescale each image to its original size
                det[:, :4] = scale_coords(batch.shape[2:], det[:, :4], shapes[i][0],shapes[i][1]).round()
               
                # convert tensor to list and scalar
                out_boxes = OrderedDict()
                boxes = []
                labels = []
                scores = []
                for *xyxy, conf, cls in det:
                    rec = torch.tensor(xyxy).view(1,4).view(-1).int().tolist()
                    conf = (conf).view(-1).detach().tolist()[0]
                    cls = (cls).view(-1).detach().tolist()[0]

                    boxes.append(rec)
                    labels.append(cls)
                    scores.append(conf)

                out_boxes['boxes'] = boxes
                out_boxes['labels'] = labels
                out_boxes['scores'] = scores   
                output[paths[i]] = out_boxes
        
        return output

if __name__ == '__main__':
    weight_path = "checkpoints/scratch_merimen/best.pt"
    img_path1 = ['test_dev/https:__s3.amazonaws.com_mc-ai_dataset_india_20190312_6_DSCN9483.JPG','test_dev/test_car3.jpeg']
    img_path2 = 'test_dev'
    img_path3 = 'test_dev/https:__s3.amazonaws.com_mc-ai_dataset_india_20190312_6_DSCN9483.JPG'
    model = Dev_model(weight_path,0.5,0.5)
    out1 = model.inference(img_path1)
    print(out1)
    print('*'*10)

    out2 = model.inference(img_path2)
    print(out2)
    print('*'*10)

    out3 = model.inference(img_path3)
    print(out3)
    print('*'*10)
