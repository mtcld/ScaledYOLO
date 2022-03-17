import torch
import argparse

def clean_checkpoint(ckpt_path,output_path):
    ckpt = torch.load(ckpt_path)
    ckpt['epoch'] = -1
    ckpt['best_fitness'] = 0
    ckpt['training_results'] = None
    ckpt['optimizer'] = None

    torch.save(ckpt,output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path',type=str,help='path to checkpoint file')
    parser.add_argument('output_path',type=str,help='path to output file')
    
    args = parser.parse_args()
    #print('check : ',args)
    ckpt_path = args.ckpt_path
    output_file = args.output_path

    clean_checkpoint(ckpt_path,output_file)



if __name__ == '__main__':
    main()