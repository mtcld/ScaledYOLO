import json
from unicodedata import category
import cv2
from tqdm import tqdm
from pathlib import Path

mode = ['merimen_v2']
damage = 'origin_crack'
annt_dir = 'coco/merimen_coco/06_04_2022/'+damage+'/annotations'
img_dir = 'coco/merimen_coco/06_04_2022/'+damage+'/images'
output_dir = 'coco_stuff_yolo/'+damage

for m in mode:
    Path(output_dir+"/"+m).mkdir(parents=True, exist_ok=True)
    with open(annt_dir+'/'+m+'.json') as f:
        data = json.load(f)
    for i in tqdm(range(len(data['images']))):
        image_id =  data['images'][i]['id']
        fn=data['images'][i]['file_name']

        #img=cv2.imread(img_dir+'/'+fn)
        #img_out_path=output_dir+'/'+m+'/'+fn
        
        img_out_path = img_dir+'/'+fn
        
        #print(img_out_path)
        #cv2.imwrite(img_out_path,img)

        with open(output_dir+'/'+m+'.txt','a') as file_text:
            file_text.write(img_out_path+'\n')

        fn_text = fn[:fn.rfind('.')]+'.txt'
        height = data['images'][i]['height']
        width = data['images'][i]['width']

        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==image_id:
                category_id = data['annotations'][j]['category_id']
                bbox = data['annotations'][j]['bbox']

                x_c = (bbox[0]+0.5*bbox[2])/width
                y_c = (bbox[1] + 0.5 * bbox[3]) / height
                x_w = bbox[2] / width
                y_h = bbox[3] / height

                if x_c > 1 :
                    x_c = 1
                
                if y_c > 1 :
                    y_c = 1
                
                if x_w > 1 :
                    x_w = 1
                
                if y_h > 1 :
                    y_h = 1

                with open(output_dir+'/'+m+'/'+fn_text, 'a') as the_file:
                    the_file.write('0 '+str(x_c)+' '+str(y_c)+' '+str(x_w)+' '+str(y_h)+'\n')

