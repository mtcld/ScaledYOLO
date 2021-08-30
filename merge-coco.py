import json
import sys

dt=json.load(open('/mmdetection/data/dent/annotations/dent_train.json'))
pp=json.load(open('/mmdetection/data/dentmerimen/dent/annotations/post_pseudo.json'))

print(type(dt['images']))
#sys.exit()

for i in range(len(dt['images'])):
    for j in range(len(dt['annotations'])):
        if dt['images'][i]['id']==dt['annotations'][j]['image_id']:
            print(dt['images'][i]['id'])
            dt['images'][i]['id']=dt['images'][i]['id']+1000000
            dt['annotations'][j]['image_id']=dt['annotations'][j]['image_id']+1000000

pp['images'].extend(dt['images'])
pp['annotations'].extend(dt['annotations'])

with open('dent_train_merged.json', 'w') as outfile:
    json.dump(pp, outfile, indent=4, ensure_ascii=False)
