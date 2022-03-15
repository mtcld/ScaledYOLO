import torch

checkpoint_path = 'checkpoints/26_8/dent/best.pt'
ckpt = torch.load(checkpoint_path)
ckpt['epoch'] = -1
ckpt['best_fitness'] = 0
ckpt['training_results'] = None
ckpt['optimizer'] = None

torch.save(ckpt,checkpoint_path[:checkpoint_path.rfind('.')]+'_pretrain'+checkpoint_path[checkpoint_path.rfind('.'):])