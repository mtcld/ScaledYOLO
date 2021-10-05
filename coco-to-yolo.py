import json
import cv2
from tqdm import tqdm
from pathlib import Path
damage_name='dent'
mode=['train','test','valid']
#mode=['total']
annt_dir='/workspace/share/thang/datasets/dent/annotations'
img_dir='/workspace/share/thang/datasets/dent/images'
output_dir='dent_dehaze'

for m in mode:
    Path("dent_dehaze/"+m).mkdir(parents=True, exist_ok=True)
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
            file_text.write(img_out_path+'\n')
        fn_text=fn[:fn.rfind('.')]+'.txt'
        height=data['images'][i]['height']
        width=data['images'][i]['width']
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==image_id:
                bbox=data['annotations'][j]['bbox']

                #x_c= str((bbox[0]+0.5*bbox[2])/width)
                #y_c = str((bbox[1] + 0.5 * bbox[3]) / height)
                #x_w = str(bbox[2] / width)
                #y_h = str(bbox[3] / height)
                x_c= str(min((bbox[0]+0.5*bbox[2])/width,1.0))
                y_c = str(min((bbox[1] + 0.5 * bbox[3]) / height,1.0))
                x_w = str(min(bbox[2] / width,1.0))
                y_h = str(min(bbox[3] / height,1.0))
                if float(x_c)>1 or float(x_w)>1 or float(y_c)>1 or float(y_h)>1:
                    print(fn)

               
                with open(output_dir+'/'+m+'/'+fn_text, 'a') as the_file:
                    the_file.write('0 '+x_c+' '+y_c+' '+x_w+' '+y_h+'\n')

