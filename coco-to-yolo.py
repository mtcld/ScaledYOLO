import json
import cv2
from tqdm import tqdm
from pathlib import Path
#damage_name='dent'
mode=['merimen_bbox']
annt_dir='/workspace/share/ai-damage-detection/damage_detection/augmentation/merimen_coco/26_8/dent/annotations'
img_dir='/workspace/share/ai-damage-detection/damage_detection/augmentation/merimen_coco/26_8/dent/images'
output_dir='dent_yolo'

for m in mode:
    Path(output_dir+"/"+m).mkdir(parents=True, exist_ok=True)
    with open(annt_dir+'/'+m+'.json') as f:
        data=json.load(f)
    for i in tqdm(range(len(data['images']))):
        image_id=  data['images'][i]['id']
        fn=data['images'][i]['file_name']
        img=cv2.imread(img_dir+'/'+fn)
        img_out_path=output_dir+'/'+m+'/'+fn
        #print(img_out_path)
        cv2.imwrite(img_out_path,img)
        with open(output_dir+'/'+m+'.txt','a') as file_text:
            #img_out_path = 'merge_crack_yolo/valid_mask/'+fn
            file_text.write(img_out_path+'\n')
        fn_text=fn[:fn.rfind('.')]+'.txt'
        height=data['images'][i]['height']
        width=data['images'][i]['width']
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==image_id:
                bbox=data['annotations'][j]['bbox']

                x_c= (bbox[0]+0.5*bbox[2])/width
                y_c = (bbox[1] + 0.5 * bbox[3]) / height
                x_w = bbox[2] / width
                y_h = bbox[3] / height

                if x_c >=1 or y_c >= 1 or x_w >= 1 or y_h >= 1 :
                    continue

                with open(output_dir+'/'+m+'/'+fn_text, 'a') as the_file:
                    the_file.write('0 '+str(x_c)+' '+str(y_c)+' '+str(x_w)+' '+str(y_h)+'\n')

