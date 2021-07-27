import torch
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
        dataset = LoadImagesBatch(img_paths,img_size=self.imgsz)

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

                    boxes.append(rec)
                    labels.append(cls)
                    scores.append(conf)
                    cropped_box=temp_img.copy()[rec[1]:rec[3],rec[0]:rec[2]]
                    cropped_box_size=cropped_box.shape
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
                    cv2.imwrite('temp/'+str(n1)+'.png',segm_thresh)
                    main_img=overlay_mask(main_img,segm_thresh,rec[0],rec[1],str(n1))
                    
                    cv2.rectangle(main_img, (rec[0], rec[1]), (rec[2], rec[3]), (255,0,0), 2)
                    
                out_boxes['boxes'] = boxes
                out_boxes['labels'] = labels
                out_boxes['scores'] = scores   
                output[paths[i]] = out_boxes
                print('outboxes')
                print(out_boxes)
                cv2.imwrite(paths[i].replace('test_dev','out'),main_img)
                        
        return output

if __name__ == '__main__':
    weight_path = "/mmdetection/data/scratch_model/best.pt"
    #weight_path = "./pytorch-unet/best.pt"
    #img_path1 = ['test_dev/car.jpeg']
    img_path1 = ['test_dev/car.jpeg','test_dev/a7.jpg']
    model = Dev_model(weight_path,0.3,0.3)
    out1 = model.inference(img_path1)
    print(out1)
    print('*'*10)

    #out2 = model.inference(img_path2)
    #print(out2)
    #print('*'*10)

  
