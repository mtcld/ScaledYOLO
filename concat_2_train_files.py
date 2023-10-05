from pathlib import Path 
import argparse
import os

def concat_files(files,output_path):
    des = output_path.replace('.txt','') 
    Path(des).mkdir(parents=True,exist_ok=True)
    
    with open(files[0]) as f :
        origin_data = [line for line in f]
        os.system('cp '+files[0].replace('.txt','')+ '/* '+des)
        

    for file in files[1:]:
        with open(file) as f :
            merimen_data = [line for line in f]

        addition = [line for line in merimen_data if line not in origin_data]
        
        print('duration :',len(merimen_data)-len(addition))
        print('addition :',len(addition))
        
        origin_data.extend(addition)
        os.system('cp '+file.replace('.txt','')+ '/* '+des)
    
    print(files[0].replace('.txt',''),files[0])

    with open(output_path,'w') as f:
        for line in origin_data:
            f.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',nargs='+',type=str,help='path to file')
    # parser.add_argument('second_file',type=str,help='path to second file')
    parser.add_argument('--output_file',type=str,help='path to output file')

    args = parser.parse_args()
    print(args)

    concat_files(args.files, args.output_file)

if __name__ == '__main__':
    main()