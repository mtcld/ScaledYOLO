
import glob
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm_notebook
import cv2
from PIL import Image
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np
import helper
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp


import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(self.base_model.children())                
        
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
    

def unet_pred(input_batch):
    base_model = models.resnet18(pretrained=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(1)
    model = model.to(device)
    model.load_state_dict(torch.load('pytorch-unet/dent_best.pt'))


    model.eval()   # Set model to evaluate mode

    

    inputs_batch= input_batch.to(device)

    print('input shape')
    print(input_batch.shape)
    input_batch=input_batch.float()
    input_batch=input_batch.cuda()
    pred = model(input_batch)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

    return pred_rgb



#read input images
damage_name='dent'
imgs_path=glob.glob('main_'+damage_name+'_test/*.png')
#imgs_path=glob.glob('test_dev/*.png')
print(len(imgs_path))
dice=[]
n=0
for i in imgs_path:
    n=n+1
    print(n)
    img=cv2.imread(i)
    gt_mask=cv2.imread(i.replace('main_',''))
    gt_mask=cv2.resize(gt_mask,(96,96)) 
    _, mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    h,w=img.shape[:2]
    img=cv2.resize(img,(96,96))
    #cropped_box=cropped_box.transpose()
    img=img.transpose()
    img=img.reshape(3,96,96)
    #img=img.transpose()
    img=torch.from_numpy(img)
    img=torch.unsqueeze(img, 0)
    img=img/255
    segm_pred=unet_pred(img)[0]
    #segm_pred=cv2.resize(segm_pred,(w,h))    
    _, pred = cv2.threshold(segm_pred, 127, 255, cv2.THRESH_BINARY)
    #pred=pred/255
    #cv2.imwrite(str(n)+'.png',pred)
    print(mask.shape,pred.shape)
    print(np.unique(mask),np.unique(pred))
    try:
        intersection = np.logical_and(mask, pred)
        if len(np.unique(pred,return_counts=True)[1])>1:
            ground=np.unique(mask,return_counts=True)[1][1]
            pred_val=np.unique(pred,return_counts=True)[1][1]
            dice_score = 2*np.sum(intersection) / (ground+pred_val)
        else:
            dice_score=0
        print(dice_score)
        dice.append(dice_score)
    except:
        print('No mask')
    
final_dice=sum(dice)/len(dice)
print(len(dice))
print('Dice Coeff: '+str(final_dice))
    
    
    
