import os
import matplotlib.pyplot as plt
import sys
import matplotlib.pylab as pylab
import cv2
import json
from io import BytesIO
import csv
import pathlib
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import namedtuple

# # Logging
import logging
# # Logging


def rect_area_intersect(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    area1=(a.xmax-a.xmin)*(a.ymax-a.ymin)
    area2=(b.xmax-b.xmin)*(b.ymax-b.ymin)
    if (dx>=0) and (dy>=0):
        area_intersection=dx*dy
        area_union=area1+area2-area_intersection
        area_ratio=area_intersection/area_union
        return area_ratio
    else:
        return 0
    
def rect_area_single(a):  # returns None if rectangles don't intersect
    dx = a.xmax- a.xmin
    dy = a.ymax- a.ymin
    return dx*dy

def size_check(path,image,r_org,damage_name):
    
    area_org=rect_area_single(r_org)
    if area_org<32**2:
        dent_path=damage_name+'/small/'
    if area_org>32**2 and area_org<96**2:
        dent_path=damage_name+'/medium/'
    if area_org>96**2:
        dent_path=damage_name+'/large/'
    path=path.replace(file_store,'')
    final_path=dent_path+path
    print(final_path)
    cv2.imwrite(final_path,image)
    
def size_check_ann(r_org):
    area_org=rect_area_single(r_org)
    if area_org<32**2:
        size='small'
    if area_org>32**2 and area_org<96**2:
        size='medium'
    if area_org>96**2:
        size='large'
    return size


damage_name='dent'


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

file_store=damage_name + '_files/'
fp_store=damage_name + '_fp/'


test_json='/mmdetection/data/dent_latest2/annotations/dent_test.json'
pred_json='detections_val2017__results.json'
img_dir='/mmdetection/data/dent_latest2/images/'

with open(test_json) as f:
    data=json.load(f)

with open(pred_json) as f:
    pred_data=json.load(f)
    
pathlib.Path(damage_name+'/small/').mkdir(parents=True, exist_ok=True) 
pathlib.Path(damage_name+'/medium/').mkdir(parents=True, exist_ok=True) 
pathlib.Path(damage_name+'/large/').mkdir(parents=True, exist_ok=True) 
pathlib.Path(file_store).mkdir(parents=True, exist_ok=True) 
pathlib.Path(fp_store).mkdir(parents=True, exist_ok=True) 


confusion_matrix={}
data_out=[]
IOU_25=0
IOU_50=0
TP25=0
TP50=0
FP=0
FN=0


    
for i in range(len(data['images'])):
    fp_check=0
    tp_temp=0
    fp_temp=0
    fn_temp=0
    
    annt_dict={}
    h=data['images'][i]['height']
    w=data['images'][i]['width']

    file_name=data['images'][i]['file_name']
    file_id=data['images'][i]['id']
    print(img_dir+file_name)
    img=cv2.imread(img_dir+file_name)
    print(img.shape)
    image_new_org=img.copy()
    org_bbox_dict={}
    
    p_org=[]
    for j in range(len(data['annotations'])):
        if data['annotations'][j]['image_id']==data['images'][i]['id']:
            #if data['annotations'][j]['category_id']!=catnew:
            #    print('continue')
            #    continue
            bbox_org=data['annotations'][j]['bbox']
            org_bbox_id=data['annotations'][j]['id']
            r_org=Rectangle(int(bbox_org[0]), int(bbox_org[1]),int(bbox_org[2]+bbox_org[0]),int(bbox_org[3]+bbox_org[1]))
            org_bbox_dict[org_bbox_id]=r_org
            cv2.rectangle(image_new_org,(r_org.xmin,r_org.ymin),(r_org.xmax,r_org.ymax),(0,255,0),2)
            p_org.append(bbox_org)
            print(file_store+file_name)
            cv2.imwrite(file_store+file_name,image_new_org)
            
            
    
    for yp in pred_data:
        if yp['image_id']==file_id and yp['score']>0.25:
            bbox_pred=yp['bbox']
            area_list=[]
            bbox_detected=[]
            p_pred=[]
            if len(org_bbox_dict.keys())==0:
                fp_check=1 
                fp_temp=fp_temp+1
    
            else:
                area_dict={}
                r_pred=Rectangle(int(bbox_pred[0]),int(bbox_pred[1]),int(bbox_pred[0]+bbox_pred[2]),int(bbox_pred[1]+bbox_pred[3]))
                print('r_pred')
                print(r_pred)
                for org_keys in org_bbox_dict.keys():
                    r_org=org_bbox_dict[org_keys]
                    area=rect_area_intersect(r_org,r_pred)
                    area_dict[org_keys]=area
                
                ann_detected=max(area_dict, key=area_dict.get)
                if ann_detected in bbox_detected:
                    continue
                if max(area_dict.values())>0.25:
                    TP25=TP25+1
                    tp_temp=tp_temp+1
                        
                    status='TP'
                    IOU_25=1
                    bbox_detected.append(ann_detected)
                        
                    r_org=org_bbox_dict[ann_detected]
                        
                    size=size_check_ann(r_org)
                        
                    cv2.rectangle(image_new_org,(r_pred.xmin,r_pred.ymin),(r_pred.xmax,r_pred.ymax),(255,0,0),2)
                        
                    p_pred.append(bbox_pred)
                        
                    path_save=file_store+file_name
                    cv2.imwrite(path_save,image_new_org)
                                                
                    if max(area_dict.values())>0.5:
                        TP50=TP50+1
                        IOU_50=1
                    else:
                        IOU_50=0
                
                
                else:
                    FP=FP+1
                    fp_temp=fp_temp+1
                    status='FP'
                    fp_check=1
                    IOU_25=0
                    IOU_50=0
                    ann_detected=-1
                    p_pred.append(bbox_pred)
                        
                    cv2.rectangle(image_new_org,(r_pred.xmin,r_pred.ymin),(r_pred.xmax,r_pred.ymax),(0,0,255),2)
                                                
                    cv2.imwrite(file_store+file_name,image_new_org)
                    size=-1
            
    for ann in org_bbox_dict.keys():
        if ann in bbox_detected:
            continue
        FN=FN+1
        fn_temp=fn_temp+1
        r_org=org_bbox_dict[ann]
        size=size_check_ann(r_org)
            
    if fp_check==1:
        cv2.imwrite(fp_store+file_name,image_new_org)
    print(len(org_bbox_dict.keys()))            
    print(tp_temp,fp_temp,fn_temp)                
confusion_matrix={'true_positve_25':TP25,'true_positve_50':TP50,'false_positive':FP, 'false_negative':FN}

# # Logging
logging.basicConfig(filename='metrics.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('true_positve_25: %s', TP25)
logging.info('true_positve_50: %s', TP50)
logging.info('false_positive: %s', FP)
logging.info('false_negative: %s', FN)
# # Logging

with open(damage_name+'_confusion_matrix.json', 'w') as outfile:
        json.dump(confusion_matrix,outfile,indent=4,ensure_ascii = False)
