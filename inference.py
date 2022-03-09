import torch
from models.experimental import attempt_load
from utils.datasets import LoadImagesBatch,LoadImageSAHI
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from collections import OrderedDict

class Scale_yolo_model():
    def __init__(self,weight_path,iou_thres=0.2,imgsz=896):
        self.device = select_device('1')
        self.half = self.device.type != 'cpu'
        #self.conf_score = conf_score
        self.iou_thres = iou_thres
        #print('debug init model :',self.iou_thres,imgsz)

        self.model = attempt_load(weight_path, map_location=self.device)  # load FP32 model
        if imgsz is not None : 
            self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        else:
            self.imgsz = check_img_size(896, s=self.model.stride.max())

        if self.half : 
            self.model.half() 

        #print(self.model.names)  
    
    def inference(self,img_paths,conf_score):
        self.conf_score = conf_score
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
        
        #print('pred : ',pred.shape,pred[0,0,])
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
    
    # change iamge size for SAHI inference
    def change_image_size(self,imgsz):
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())

    def inference_ndarray(self,image,conf_score=0.1):
        self.conf_score = conf_score
        #print('debug imgsz: ',self.imgsz)
        dataset = LoadImageSAHI(image,self.imgsz)

        for img, img0, shape in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            #print('debug img0 shape : ',img0.shape)

        # run batch inference
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

        # Apply MNS
        #print('debug nms : ',self.conf_score, self.iou_thres)
        pred = non_max_suppression(pred, self.conf_score, self.iou_thres, agnostic=True)
        
        # batch with only 1 image
        det = pred[0]

        boxes = []
        labels = []
        scores = []
        #for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # remove batch padding and rescale each image to its original size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], shape[0],shape[1]).round()
            
            # convert tensor to list and scalar
            
            for *xyxy, conf, cls in det:
                rec = torch.tensor(xyxy).view(1,4).view(-1).int().tolist()
                conf = (conf).view(-1).detach().tolist()[0]
                cls = (cls).view(-1).detach().tolist()[0]

                boxes.append(rec)
                labels.append(cls)
                scores.append(conf)
        
        return [[boxes,scores,labels]]

if __name__ == '__main__':
    weight_path = "scratch.pt"
    img_path1 = ['https:__s3.amazonaws.com_mc-ai_dataset_india_20190312_1_IMG_20170417_101024.jpg']
    #img_path2 = 'test_dev'
    #img_path3 = 'test_dev/https:__s3.amazonaws.com_mc-ai_dataset_india_20190312_6_DSCN9483.JPG'
    model = Scale_yolo_model(weight_path,0.5,0.5)
    out1 = model.inference(img_path1)
    #print(out1)
    #print('*'*10)

    #out2 = model.inference(img_path2)
    #print(out2)
    #print('*'*10)

    #out3 = model.inference(img_path3)
    #print(out3)
    #print('*'*10)
