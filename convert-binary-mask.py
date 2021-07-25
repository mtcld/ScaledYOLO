from tqdm import tqdm
import cv2
import json
from pathlib import Path
import numpy as np

data_path='/mmdetection/data/scratch/'
data_name='scratch'
mode=['train','test','valid']
img_dir=data_path+'/images/'
#path = Path("parentdirectory/mydirectory")
#path.mkdir(parents=True, exist_ok=True)

for m in mode:
    path = Path(data_name+'_'+m)
    print(path)
    path.mkdir(parents=True, exist_ok=True)
    path=Path('main_'+data_name+'_'+m)
    path.mkdir(parents=True, exist_ok=True)

    with open(data_path+'annotations/'+data_name+'_'+m+'.json') as f:
        data=json.load(f)
    for i in tqdm(range(len(data['images']))):
        image_id=  data['images'][i]['id']
        fn=data['images'][i]['file_name']
        img=cv2.imread(img_dir+'/'+fn)
        #print(img.shape)
        #img_out_path=output_dir+'/'+m+'/'+fn
        #print(img_out_path)
        #cv2.imwrite(img_out_path,img)
        #with open(output_dir+'/'+m+'.txt','a') as file_text:
        #    file_text.write(img_out_path+'\n')
        #fn_text=fn[:fn.rfind('.')]+'.txt'
        height=data['images'][i]['height']
        width=data['images'][i]['width']
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==image_id:
                mask=np.zeros((height,width),dtype='uint8')
                bbox=data['annotations'][j]['bbox']
                bbox=[int(i) for i in bbox]
                p1=data['annotations'][j]['segmentation'][0]

                p1=[int(i) for i in p1]
                p2=[]
                for p in range(int(len(p1)/2)):
                    p2.append([p1[2*p],p1[2*p+1]])
                fill_pts = np.array([p2], np.int32)
                cv2.fillPoly(mask, fill_pts, 255)
                mask_new=mask.copy()[bbox[1]:bbox[3]+bbox[1],bbox[0]:bbox[2]+bbox[0]]
                img_new=img.copy()[bbox[1]:bbox[3]+bbox[1],bbox[0]:bbox[2]+bbox[0]]
                #print(bbox)
                #print(data_name+'_'+m+'/'+str(i)+'.png')
                #print(mask_new.shape)
                if mask_new.shape[0]*mask_new.shape[1]<35*35 or  mask_new.shape[0]==0 or mask_new.shape[1]==0:
                    continue
                print('mask new shape')
                print(mask_new.shape)
                mask_new=cv2.resize(mask_new,(256,256))
                ret, thresh1 = cv2.threshold(mask_new, 127, 255, cv2.THRESH_BINARY)
                #thresh1=thresh1/255
                print(np.unique(thresh1))
                cv2.imwrite(data_name+'_'+m+'/'+str(i)+'.png',thresh1)
                #img_new=cv2.resize(img_new,(256,256))
                #ret, thresh1 = cv2.threshold(mask_new, 127, 255, cv2.THRESH_BINARY)
                #thresh1=thresh1/255
                #print(np.unique(thresh1))
                img_new=cv2.resize(img_new,(256,256))
                cv2.imwrite('main_'+data_name+'_'+m+'/'+str(i)+'.png',img_new)
