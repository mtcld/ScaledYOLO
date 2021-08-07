import sys
import torch
from tqdm import tqdm
import json
import glob
import cv2
import unet_inference
import numpy as np
from unet_inference import unet_pred
from models.experimental import attempt_load
from utils.datasets import LoadImagesBatch
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from collections import OrderedDict
from PIL import Image

def convertImage(img):
    img = Image.fromarray(img)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    im_np = np.asarray(img)
    print('ss')
    print(im_np.shape)
    return im_np

def overlay_image_alpha(img, img_overlay, x, y):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    alpha_mask = img_overlay[:, :, 3] / 255.0
    
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    
    return img_crop

def overlay_mask(img,mask,x1,x2,n1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA).copy()
    red_mask=np.zeros((mask.shape[0],mask.shape[1],3),np.uint8)
    red_mask[:,:]=(0,0,255)
    masked = cv2.bitwise_and(red_mask, mask)
    #cv2.imwrite('temp/'+str(n1)+'.png',masked)
    
#     h, w, d = masked.shape #(217, 232, 3)
#     nr_of_new_layers = 1
#     alpha_img = np.zeros((h, w, d+nr_of_new_layers))
#     alpha_img[:,:,:3] = masked  #alpha_img shape = (217, 232, 4)
#     alpha_img[np.where(alpha_img[:,:,2])==255][:,:,4] = 255  #alpha_img shape = (217, 232, 4)
#     print('a')
#     print(alpha_img)
#     print(np.unique(alpha_img[:,:,2]))
    alpha_img=convertImage(masked)
    cv2.imwrite('temp/'+str(n1)+'.png',alpha_img)
    
    print('alpha shape')
    print(alpha_img.shape)

    overlay_image_alpha(img,alpha_img,x1,x2)
    return img
    

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
        z1=0
        # create dataset
        print('First step')
        dataset = LoadImagesBatch(img_paths,img_size=self.imgsz)
        #sys.exit()
        # load dataset and create batch with size equal number of given img_paths
        batch = []
        shapes = []
        paths = []
        n=0
        for path, img, im0s, shape in dataset:
            n=n+1
            print(path)
            #main_img=cv2.imread(path)
            #cv2.imwrite(path.replace('test_dev','out'),main_img)
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
        #print(pred)
        pred = non_max_suppression(pred, self.conf_score, self.iou_thres, agnostic=True)
        #print(pred)
        #print('Done')
        
        # Post process
        n1=0
        output = OrderedDict()
        for i, det in enumerate(pred):  # detections per image
            main_img=cv2.imread(paths[i])
            h,w=main_img.shape[:2]
            print('wh')
            print(h,w)
            mask_img=np.zeros((h,w),np.uint8)
            temp_img=main_img.copy()
            print(paths[i])
            if det is not None and len(det):
                #print(det)
                # remove batch padding and rescale each image to its original size
                det[:, :4] = scale_coords(batch.shape[2:], det[:, :4], shapes[i][0],shapes[i][1]).round()
                
                # convert tensor to list and scalar
                out_boxes = OrderedDict()
                boxes = []
                labels = []
                scores = []
                
                for *xyxy, conf, cls in det:
                    n1=n1+1
                    rec = torch.tensor(xyxy).view(1,4).view(-1).int().tolist()
                    conf = (conf).view(-1).detach().tolist()[0]
                    cls = (cls).view(-1).detach().tolist()[0]
                    rec[2]=min(w,rec[2])
                    rec[3]=min(h,rec[3])
                    
                    print('re')
                    print(rec)
                    
                    boxes.append(rec)
                    labels.append(cls)
                    scores.append(conf)
                    cropped_box=temp_img.copy()[rec[1]:rec[3],rec[0]:rec[2]]
                    cropped_box_size=cropped_box.shape
                    print('c shape')
                    print(cropped_box_size)
                    cropped_box=cv2.resize(cropped_box,(96,96))
                    z1=z1+1
                    cv2.imwrite('cropped/'+str(z1)+'.png',cropped_box)
                    cropped_box=cropped_box.transpose()
                    cropped_box=cropped_box.reshape(3,96,96,order = 'C')
                    #cropped_box=cropped_box.transpose((0,,1))
                    cropped_box=torch.from_numpy(cropped_box)
                    cropped_box=cropped_box/255
                    cropped_box=torch.unsqueeze(cropped_box, 0)
                    
                    
                    segm_pred=unet_pred(cropped_box)[0]
                    cv2.imwrite('cropped/s'+str(z1)+'.png',segm_pred)
                    #segm_pred=segm_pred.T.reshape((96,96,3))
                    
                    segm_pred=cv2.resize(segm_pred,(cropped_box_size[1],cropped_box_size[0]))
                    
                    _, segm_thresh = cv2.threshold(segm_pred, 127, 255, cv2.THRESH_BINARY)
                    print(np.unique(segm_thresh))
                    
                    

                    #if rec[3]-rec[1]==0  or rec[2]-rec[0]==0:
                    #    continue
                    segm_thresh = cv2.cvtColor(segm_thresh, cv2.COLOR_BGR2GRAY)
                    print('sts')
                    print(segm_thresh.shape)
                    print('mask')
                    print(mask_img.shape)
                    cv2.imwrite('temp/'+str(n1)+'.png',segm_thresh)
                    print(mask_img[rec[1]:rec[3],rec[0]:rec[2]].shape)
                    #main_img=overlay_mask(main_img,segm_thresh,rec[0],rec[1],str(n1))
                    m_shape=mask_img[rec[1]:rec[3],rec[0]:rec[2]].shape
                    if m_shape[0]*m_shape[1]==0:
                       continue
                    mask_img[rec[1]:rec[3],rec[0]:rec[2]]=segm_thresh
                    
                    #cv2.rectangle(main_img, (rec[0], rec[1]), (rec[2], rec[3]), (255,0,0), 2)
                    
                out_boxes['boxes'] = boxes
                out_boxes['labels'] = labels
                out_boxes['scores'] = scores   
                output[paths[i]] = out_boxes
                print('outboxes')
                print(out_boxes)
                print('output/'+paths[i][path[i].rfind('/')+1:])
                #print(main_img[:,:,0:3].shape,temp_img.shape)
                #im_h = cv2.hconcat([temp_img, mask_img])
                cv2.imwrite('maskscratch/'+paths[i][paths[i].rfind('/')+1:],mask_img)
                        
        return mask_img

if __name__ == '__main__':
    damage_name='scratch'
    weight_path = '/mmdetection/data/'+damage_name+'_model/best.pt'
    print(weight_path)
    #weight_path = "./pytorch-unet/best.pt"
    #img_path1 = ['test_dev/car.jpeg']
    #img_path1 = ['test_dev/car.jpeg','test_dev/a7.jpg']
    img_path1=[]
    json_path='/mmdetection/data/'+damage_name+'/annotations/'+damage_name+'_test.json'
    data=json.load(open(json_path))
    model = Dev_model(weight_path,0.3,0.3)
    dice=[]
    l=0
    for i in tqdm(range(len(data['images']))):
        try:
            h=data['images'][i]['height']
            w=data['images'][i]['width']
            print(h,w)
            mask=np.zeros((h,w),dtype='uint8')
            for j in range(len(data['annotations'])):
                if data['annotations'][j]['image_id']==data['images'][i]['id']:
                    p1=data['annotations'][j]['segmentation'][0]
                    p1=[int(i) for i in p1]
                    p2=[]
                    for p in range(int(len(p1)/2)):
                        p2.append([p1[2*p],p1[2*p+1]])
                    fill_pts = np.array([p2], np.int32)
                    cv2.fillPoly(mask, fill_pts, 1)
            #print(np.unique(mask))
            #print(np.unique(mask,return_counts=True)[1][1]/(w*h))
            if np.unique(mask,return_counts=True)[1][1]/(w*h)>0.000:
                img_path1=('/mmdetection/data/'+damage_name+'/images/'+data['images'][i]['file_name'])
                img=cv2.imread(img_path1)
                
                pred = model.inference(img_path1)
                pred=pred/255
                #out = predictor(img)
                #pred = torch.sum(out['instances'].pred_masks,dim=0) > 0
                #pred = pred.cpu().detach().numpy()
                pred=pred.astype(int)
                intersection = np.logical_and(mask, pred)
                if len(np.unique(pred,return_counts=True)[1])>1:
                    ground=np.unique(mask,return_counts=True)[1][1]
                    pred_val=np.unique(pred,return_counts=True)[1][1]
                    dice_score = 2*np.sum(intersection) / (ground+pred_val)
                else:
                    dice_score=0
                dice.append(dice_score)
        except Exception as e:
            print(e)
    final_dice=sum(dice)/len(dice)
    print('Dice Coeff: '+str(final_dice))
