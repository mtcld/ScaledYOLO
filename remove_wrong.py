from pathlib import Path 
import random
import argparse


# wrong_path  = 'coco_stuff_yolo/new_scratch/gen_train_crop.txt'
# origin_path = 'coco_stuff_yolo/new_scratch/merge_train.txt'

def remove(origin_path,wrong_path):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_path',type=str,help='path to label file')
    parser.add_argument('wrong_path',type=str,help='path to wrong files need to remove')
    
    args = parser.parse_args()
    #print('check : ',args)
    origin_path = args.label_path
    wrong_path = args.wrong_path

    remove(origin_path,wrong_path)



if __name__ == '__main__':
    main()