filename='train_aug.txt'

with open(filename) as f:
    content = f.readlines()

print(len(content))

print(content[0])


for c in content:
    if 's3.amazonaws.com_mc-ai_dataset_india_20190312_253_mark+(4)' not in c:
         with open(filename, 'a') as the_file:
             the_file.write(c+'\n')
