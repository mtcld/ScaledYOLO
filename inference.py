import torch
from models.experimental import attempt_load
from utils.datasets import LoadImagesBatch
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from collections import OrderedDict
import cv2

class Dev_model():
    def __init__(self,weight_path,iou_thres=0.5,label=0,imgsz = 896):
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        #self.conf_score = conf_score
        self.iou_thres = iou_thres
        self.label = label

        self.model = attempt_load(weight_path, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size

        if self.half : 
            self.model.half()   
    
    def inference(self,img_paths,conf_score,flip=False):
        # create dataset
        self.conf_score = conf_score
        dataset = LoadImagesBatch(img_paths,img_size=self.imgsz,flip=flip)

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

        #print('debug:',pred.shape )
        # Apply MNS
        pred = non_max_suppression(pred, self.conf_score, self.iou_thres, agnostic=False)

        # Post process
        output = OrderedDict()
        for i, det in enumerate(pred):  # detections per image
            out_boxes = OrderedDict()
            boxes = []
            labels = []
            scores = []
            if det is not None and len(det):
                # remove batch padding and rescale each image to its original size
                det[:, :4] = scale_coords(batch.shape[2:], det[:, :4], shapes[i][0],shapes[i][1]).round()
               
                # convert tensor to list and scalar
                
                
                for *xyxy, conf, cls in det:
                    h,w = shapes[i][0]
                    rec = torch.tensor(xyxy).view(1,4).view(-1).int().tolist()

                    if flip : 
                        rec[0] = w - rec[0]
                        rec[2] = w - rec[2]

                        #rec = [rec[2],rec[3],rec[0],rec[1]]

                    conf = (conf).view(-1).detach().tolist()[0]
                    #cls = (cls).view(-1).detach().tolist()[0]
                    h,w = shapes[i][0]
                    #new_rec = [[rec[0]/w,rec[1]/h],[rec[2]/w,rec[1]/h],[rec[2]/w,rec[3]/h],[rec[0]/w,rec[3]/h]]
                    #print(rec)
                    #print(new_rec)
                    #print('*'*10)
                    boxes.append(rec)
                    labels.append(self.label)
                    scores.append(conf)

            out_boxes['boxes'] = boxes
            out_boxes['labels'] = labels
            out_boxes['scores'] = scores   
            output[paths[i]] = out_boxes
        
        return output

if __name__ == '__main__':
    weight_path = "checkpoints/scratch/scratch_23_9.pt"
    img_path1 = ['coco/scratch/images/https:__s3.amazonaws.com_mc-imt_vehicle_2019Y7149_vehicle_additional_docs_21749_medium_15559750008971833159462424792505.jpg']
    model = Dev_model(weight_path,0.3,0.5,'scratch')
    #import time
    out1 = model.inference(img_path1,flip=True)
    print(out1)
    img = cv2.imread(img_path1[0])
    for b in out1[img_path1[0]]['boxes'] :
        cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(255,0,0),2)
    cv2.imwrite('demo.png',img)