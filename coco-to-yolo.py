import json
import cv2
from tqdm import tqdm
from pathlib import Path
import numpy as np
damage_name='missing'
mode=['train','test','valid']
#mode=['val']
annt_dir='/mmdetection/data/total-missing/annotations'
img_dir='/mmdetection/data/total-missing/images'
output_dir='data5'

for m in mode:
    Path("data5/"+m).mkdir(parents=True, exist_ok=True)
    with open(annt_dir+'/'+damage_name+'_'+m+'.json') as f:
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
        fn_json=fn[:fn.rfind('.')]+'.json'
        height=data['images'][i]['height']
        width=data['images'][i]['width']
        out_list=[]
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==image_id:
                bbox=data['annotations'][j]['bbox']
                #print(type(data['annotations'][j]['segmentation']))
                if type(data['annotations'][j]['segmentation']) is dict:
                    print('continue')
                    continue
                segm=data['annotations'][j]['segmentation'][0]
                #print('bbox')
                #print(bbox)
                #print('segm')
                #print(segm)
                pnt=[]
                for pp in range(int(len(segm)/2)):
                    pnt.append([int(segm[2*pp]),int(segm[2*pp +1])])
                #print('pnt')
                #print(pnt)

                a3=np.array([pnt])
                mask=np.zeros([width,height],dtype=np.uint8)
                cv2.fillPoly( mask, a3, 255 )
                mask=cv2.resize(mask,(56,56))
                retval, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                thresh=thresh/255
                thresh = thresh.astype('uint8')
                #print(np.unique(thresh))
                #print('thresh')
                #print(thresh)
                out_list.append(thresh.tolist())


                x_c= str((bbox[0]+0.5*bbox[2])/width)
                y_c = str((bbox[1] + 0.5 * bbox[3]) / height)
                x_w = str(bbox[2] / width)
                y_h = str(bbox[3] / height)


                with open(output_dir+'/'+m+'/'+fn_text, 'a') as the_file:
                    the_file.write('0 '+x_c+' '+y_c+' '+x_w+' '+y_h+'\n')
        out_json={'segm':out_list}
        with open(output_dir+'/'+m+'/'+fn_json, 'w') as outfile:
            json.dump(out_json,outfile,indent=4,ensure_ascii = False)
