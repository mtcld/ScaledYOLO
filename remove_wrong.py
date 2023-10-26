from pathlib import Path 
import random

wrong_path  = 'coco_stuff_yolo/new_scratch/gen_train_crop.txt'
origin_path = 'coco_stuff_yolo/new_scratch/merge_train.txt'

wrong = []
with open(wrong_path) as f:
    for line in f.readlines():
        wrong.append(Path(line).stem)

print(wrong)
# wrong = random.sample(wrong,int(0.5*len(wrong)))

data = []
with open(origin_path,'r') as f:
    for line in f.readlines():
        data.append(line)

new_data = [i for i in data if Path(i).stem not in wrong]

# print(data[:3])
print(len(data),len(new_data))

with open(origin_path,'w') as f:
    for line in new_data:
        f.write(line)