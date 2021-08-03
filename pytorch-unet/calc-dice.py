import glob
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import helper
import simulation


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = models.resnet18(pretrained=False)
# class SimDataset(Dataset):
#     def __init__(self, count, transform=None):
#         self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)        
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.input_images)
    
#     def __getitem__(self, idx):        
#         image = self.input_images[idx]
#         mask = self.target_masks[idx]
#         if self.transform:
#             image = self.transform(image)
        
#         return [image, mask]

class SimDataset(Dataset):
    def __init__(self,  transform, subset="train",damage_name='scratch'):
        super().__init__()
        #self.df = df
        self.transform = transform
        self.subset = subset
        self.damage_name=damage_name
        self.fn=glob.glob('../main_'+self.damage_name+'_'+self.subset+'/*.png')
        #if self.subset == "train":
        #    self.data_path = path + 'train_images/'
        #elif self.subset == "test":
        #    self.data_path = path + 'test_images/'

    def __len__(self):
        return len(self.fn)
    
    def __getitem__(self, index):  
        
        #fn = self.df['ImageId_ClassId'].iloc[index].split('_')[0]         
        #img = Image.open( self.fn[index])
        img = cv2.imread( self.fn[index])
        #print(img.shape)
        img = self.transform(img)
        #print(img.shape)

        #if self.subset == 'train' or self.subset == 'valid':
        if 1>0:
            mask = cv2.imread(self.fn[index].replace('main_',''),0)
            #print(mask.shape)
            mask = self.transform(mask)
            #print(mask.shape)
            return img, mask
        #else: 
        #    mask = None
        #    return img, mask



# use same transform for train/val for this example
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96,96))
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])


import torch.nn as nn

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out


from collections import defaultdict
import torch.nn.functional as F
import torch
from loss import dice_loss

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    state_dict = model.state_dict()
    torch.save(state_dict, 'best5.pt')
    return model


import math

model = ResNetUNet(1)
model = model.to(device)
model.load_state_dict(torch.load('crack_best.pt'))

model.eval()   # Set model to evaluate mode
damage_name='crack'
test_dataset = SimDataset(subset="test", transform = trans,damage_name=damage_name)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)

dice=[]
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)   
    pred = model(inputs)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
    pred_rgb = [helper.masks_to_colorimg(x) for x in pred]
    
    for k in range(len(target_masks_rgb)):
        try:
            target_masks_rgb_mask=cv2.cvtColor(target_masks_rgb[k],cv2.COLOR_BGR2GRAY)
            pred_rgb_mask=cv2.cvtColor(pred_rgb[k],cv2.COLOR_BGR2GRAY)
            intersection = np.logical_and(target_masks_rgb_mask, pred_rgb_mask)
            ground=np.unique(target_masks_rgb_mask,return_counts=True)[1][1]
            pred_val=np.unique(pred_rgb_mask,return_counts=True)[1][1]
            dice_score = 2*np.sum(intersection) / (ground+pred_val)
            dice.append(dice_score)
            print(dice_score)
        except:
            print('No Mask')
    #print(dice)
print(len(dice))
print(sum(dice)/len(dice))
