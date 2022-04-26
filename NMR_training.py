
import glob
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import torch
import logging
import sys
import tempfile
import re
import time
from glob import glob
from torch import nn
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, SmartCacheDataset
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet, BasicUNet, BasicUnet_3levels, BasicUnet_4levels
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, CropForegroundd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandScaleIntensityd, RandAffined,RandSpatialCropd, AsDiscrete, Activations)
from monai.transforms.compose import MapTransform, Randomizable
from monai.utils import first, set_determinism
import numpy as np
from scipy import ndimage
from collections.abc import Iterable
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import random
from monai.config import KeysCollection

def main(temp):
    
    root_dir_res= '../..' # Results directory
    root_dir= '../..' # Models directory
    path = '../..' # Data directory

    mp2rage = sorted(glob(os.path.join(path, "*/MP2RAGE.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    les = sorted(glob(os.path.join(path, "*/gt_two.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))
    les_wm = sorted(glob(os.path.join(path, "*/mask_wm.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))

    N=60 #Number of MRI volumes
    np.random.seed(seed=111) #This can be changed to create an ensemble of models.
    indices = np.random.permutation(N)
    r = 10 #Validation samples
    v=indices[:r]
    t=indices[r:]
    
    train_files=[]
    val_files=[]
    for i in t:
        train_files = train_files + [{"les": les,"mp2rage": mp, "les_wm": les_wm} for les, mp, 
                                     les_all, les_wm in zip(les[i:i+1], mp2rage[i:i+1], les_wm[i:i+1])]
    for j in v:
        val_files = val_files + [{"les": les,"mp2rage": mp} for les, mp
                                 in zip(les[j:j+1], mp2rage[j:j+1])]
    
    train_transforms = Compose(
    [
        LoadNiftid(keys=["les", "mp2rage","les_wm"]),
        AddChanneld(keys=["les", "mp2rage", "les_wm"]),
        Spacingd(keys=["les","mp2rage", "les_wm"], pixdim=(0.5, 0.5, 0.5), mode=("nearest",
                        "bilinear", "nearest")), 
        
        NormalizeIntensityd(keys=["mp2rage"],nonzero=True),
        RandShiftIntensityd(keys=["mp2rage"],offsets=0.1,prob=1.0),
        RandScaleIntensityd(keys=["mp2rage"], factors=0.1, prob=1.0),
        
        ConcatItemsd(keys=["mp2rage"], name="image"),
        RandCropByPosNegLabeld(
            keys=["image", "les", "les_wm"],label_key="les",
            spatial_size=(96,96,96),pos=15,neg=1,num_samples=16,image_key="image"),
        
        RandFlipd (keys=["image", "les", "les_wm"],prob=0.5,spatial_axis=(0,1,2)),
        RandRotate90d (keys=["image", "les", "les_wm"],prob=0.5,spatial_axes=(0,1)),
        RandRotate90d (keys=["image", "les", "les_wm"],prob=0.5,spatial_axes=(1,2)),
        RandRotate90d (keys=["image", "les", "les_wm"],prob=0.5,spatial_axes=(0,2)),

        RandAffined(keys=['image', 'les', 'les_wm'], mode=('bilinear', 'nearest', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
                     rotate_range=(np.pi/8, np.pi/8, np.pi/8), scale_range=(0.1, 0.1, 0.1),padding_mode='border'),
        ToTensord(keys=["image", "les", "les_wm"])
    ]
    )
 
    val_transforms = Compose(
    [
        LoadNiftid(keys=["les", "mp2rage"]),
        AddChanneld(keys=["les", "mp2rage"]),
        Spacingd(keys=["les","mp2rage"], pixdim=(0.5, 0.5, 0.5),  mode=('nearest','bilinear')), 
        NormalizeIntensityd(keys=["mp2rage"],nonzero=True),
        ToTensord(keys=["mp2rage", "les"]),
    ]
    )

    #%%
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.2, num_workers=16)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,num_workers=16)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.05, num_workers=0)
    val_train_ds = CacheDataset(data=train_files, transform=val_transforms, cache_rate=0.05, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    val_train_loader = DataLoader(val_train_ds, batch_size=1, num_workers=0)
    device = torch.device("cuda") #Select device
    
    model = BasicUnet_4levels(dimensions=3,in_channels=1,out_channels=3,
                    features=(16, 32, 64, 128, 16)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), 0.5e-4)
    epoch_num = 100
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    metric_values_train = list()
    #model.load_state_dict(torch.load(os.path.join(root_dir, "../.."))) Whether to load a pre-trained model
    
    act = Activations(softmax=True)
    for epoch in range(epoch_num):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        
        step = 0
        for batch_data in train_loader:
            n_samples = batch_data["mp2rage"].size(0)
            b = 1
            for m in range(0,batch_data["image"].size(0),b):
              step += 1
              r = random.random()
              if  r> 0.50 :
                  inputs, les, les_wm = (
                    batch_data["image"][m:(m+b),0].to(device),
                    batch_data["les"][m:(m+b)].type(torch.LongTensor).to(device),
                    batch_data["les_wm"][m:(m+b)].type(torch.LongTensor).to(device))
              else:       
                  inputs, les, les_wm = (
                    batch_data["image"][m:(m+b),1].to(device),
                    batch_data["les"][m:(m+b)].type(torch.LongTensor).to(device),
                    batch_data["les_wm"][m:(m+b)].type(torch.LongTensor).to(device))   
              inputs = torch.unsqueeze(inputs, 1)
              optimizer.zero_grad()
              outputs = model(inputs)
              ce_loss = nn.CrossEntropyLoss(reduction='none')
              les[les==2]=0
              les[les>0]=1
                           
              ce = ce_loss((outputs),torch.squeeze(les,dim=1))
              gamma = 2.0
              pt = torch.exp(-ce)
              f_loss = 1*(1-pt)**gamma * ce
              loss1=f_loss
              w= (les*4)+1

              loss = torch.mean(loss1*torch.squeeze(w))
              les = torch.squeeze(les)
              outputs2 = outputs[:,3:,:,:,:]

              loss.backward()
              optimizer.step()
              epoch_loss += loss.item()
              torch.cuda.empty_cache()
              if step%1 == 0:
                  step_print = int(step/2)
                  print(f"{step_print}/{(len(train_ds)*n_samples) // (train_loader.batch_size*2)}, train_loss: {loss.item():.4f}")
              del loss, loss1, w

        epoch_loss /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            metric_sum = 0.0
            metric_count = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    inputs, les = (
                    batch_data["mp2rage"].to(device),
                    batch_data["les"].type(torch.LongTensor).to(device))
                    
                    roi_size = (108, 108, 108)
                    sw_batch_size = 2 
                    outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model,mode='gaussian')
                    les = np.squeeze(les.cpu().numpy())
                    les_wm = np.zeros_like(les)
                    les_wm[les==2]=1
                    les[les>0]=1
                    outputs2 = act(outputs).cpu().numpy()
                    seg= np.squeeze(outputs2[:,1,:,:,:]+outputs2[:,2,:,:,:]*0.1)
                    th = 0.5 
                    seg[seg>th]=1
                    seg[seg<th]=0
            
                    gt=les
                    gt[gt>0]=1
                    gt = np.squeeze(gt)    
                    value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt) + 0.0001)
                    metric_count += 1 
                    metric_sum += value
                metric = metric_sum / metric_count
                metric_values.append(metric)
                
                metric_sum_train = 0.0
                metric_count_train = 0
                for train_data in val_train_loader:
                    train_inputs, train_labels = (
                                train_data["mp2rage"].to(device),
                                train_data["les"].to(device),
                                )
                    train_outputs = sliding_window_inference(train_inputs, roi_size, sw_batch_size, model,mode='gaussian')                 
                    train_labels =train_labels.cpu().numpy()
                    train_outputs = act(train_outputs).cpu().numpy()
                    seg= np.squeeze(train_outputs[:,1,:,:,:]+train_outputs[:,2,:,:,:])
                    seg[seg>0.5]=1
                    seg[seg<0.5]=0
                    
                    gt=train_labels
                    gt[gt==2]=0
                    gt[gt>0]=1
                    gt = np.squeeze(gt)
                    
                    value_train = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt) + 0.0001)
                    print(value_train)
                    metric_count_train += 1 
                    metric_sum_train += value_train
                metric_train = metric_sum_train / metric_count_train
                metric_values_train.append(metric_train)
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_model.pth")) #Save best model
                    print("saved new best metric model")
                print("current mean dice: ",metric)   

                plt.figure("train", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Epoch Average Train Loss")
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.subplot(1, 2, 2)
                plt.title("Val and Train Mean Dice")
                x = [val_interval * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                y1 = metric_values_train
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.plot(x, y1)
                plt.show()
                plt.savefig(root_dir+'metrics.png') #Save metrics plot
                
#%%
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)