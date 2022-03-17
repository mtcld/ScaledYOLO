# Set previous checkpoint as pretrain model
## clean all information related to previous training progress
```
python make_pretrain_checkpoint.py path_to_wanted_checkpoint path_to_output_checkpoint
```

Example : 

```
python make_pretrain_checkpoint.py checkpoints/26_8/dent/best.pt checkpoints/26_8/dent/best_pretrain.pt
```

## train with new pretrain checkpoint
```
    python -m torch.distributed.launch --nproc_per_node 3 train.py --batch-size 9 --img 1024 1024 --epochs 20 --hyp data/hyp.pretrain.yaml --data data/coco.yaml --cfg models/yolov4-p7.yaml --weights 'checkpoints/26_8/dent/best_pretrain.pt' --adam --sync-bn --device 0,1,2 --name dent_merimen_19_02
```