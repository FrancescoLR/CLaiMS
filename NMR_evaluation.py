
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
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet, BasicUNet, BasicUnet_4levels
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, Activations)
from monai.transforms.compose import MapTransform, Randomizable    
from monai.utils import first, set_determinism
from monai.data import write_nifti, create_file_basename, NiftiDataset
import numpy as np
from scipy import ndimage
from collections.abc import Iterable
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from monai.config import KeysCollection

def ltpr(seg,gt,threshold=0.5,l_min=2):
    
    seg[seg>threshold]=1
    seg[seg<threshold]=0
    seg=np.squeeze(seg)
    gt=np.squeeze(gt)
    labeled_seg, num_labels = ndimage.label(seg)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(seg,labeled_seg,label_list,np.sum,float, 0)
    seg2 = np.zeros_like(seg)
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(labeled_seg == l), axis=1)
            seg2[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1
    seg=np.copy(seg2) 
    labeled_gt, num_labels = ndimage.label(gt)
    label_list = np.unique(labeled_gt)
    num_elements_by_lesion = ndimage.labeled_comprehension(gt,labeled_gt,label_list,np.sum,float, 0)
    count_ltpr=0
    if num_labels>0:
        for j in range (1,num_labels+1):
            one_lesion = np.zeros_like(labeled_gt)
            one_lesion[labeled_gt==j]=1
            if np.sum(one_lesion*seg)>0:
                count_ltpr+=1
    return (count_ltpr,num_labels)
  
def lfpr(seg,gt,les_wm, threshold=0.5,l_min=2):

    seg[seg>threshold]=1
    seg[seg<threshold]=0
    seg=np.squeeze(seg)
    gt=np.squeeze(gt)
    les_wm=np.squeeze(les_wm)
    labeled_seg, num_labels = ndimage.label(seg)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(seg,labeled_seg,label_list,np.sum,float, 0)
    seg2 = np.zeros_like(seg)
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(labeled_seg == l), axis=1)
            seg2[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1
    seg=np.copy(seg2) 
    labeled_gt, num_labels = ndimage.label(seg)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(seg,labeled_seg,label_list,np.sum,float, 0)
    count_lfpr=0
    if num_labels>0:
        for j in range (1,num_labels+1):
            one_lesion = np.zeros_like(labeled_seg)
            one_lesion[labeled_seg==j]=1
            if ((np.sum(one_lesion*gt)==0) and (np.sum(one_lesion*les_wm)==0)):
                    count_lfpr+=1
    return (count_lfpr,num_labels)  

def main(temp):
    
    root_dir_res= '../..' # Results directory
    root_dir= '../..' # Models directory
    path = '../..' # Data directory

    mp2rage = sorted(glob(os.path.join(path, "*/MP2RAGE.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    les = sorted(glob(os.path.join(path, "*/gt.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))
    les_all = sorted(glob(os.path.join(path, "*/mask_wm.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))

    N = 60 #Number of MRI volumes
    np.random.seed(seed=111) #This can be changed to create an ensemble of models.
    indices = np.random.permutation(N)
    r = 60 #Evaluation samples
    v=indices[:r]
    val_files=[]

    for j in v: #range(50,60):
        val_files = val_files + [{"les": les,"mp2rage": mp, "les_all": les_all} for les, mp, les_all 
                                 in zip(les[j:j+1], mp2rage[j:j+1],  les_all[j:j+1])]

    val_transforms = Compose(
    [
        LoadNiftid(keys=["les", "mp2rage", "les_all"]),
        AddChanneld(keys=["les", "mp2rage", "les_all"]),
        Spacingd(keys=["les","mp2rage", "les_all"], pixdim=(0.5, 0.5, 0.5), mode=("nearest",
                        "bilinear","nearest")),
        
        NormalizeIntensityd(keys=["mp2rage"],nonzero=True),
        ToTensord(keys=["mp2rage", "les", "les_all"])])
  
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1, num_workers=24)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=24)
    device = torch.device("cuda:0")
    model = BasicUnet_4levels(dimensions=3,in_channels=1,out_channels=3,
                    features=(16, 32, 64, 128, 16)).to(device)
    act = Activations(softmax=True)

    subject=0
    metric_count=0
    metric_sum=0
    les_all=0
    les_all_fpr=0
    ltpr_all = 0
    lfpr_all=0
    dice_all = 0
    model.load_state_dict(torch.load(os.path.join(root_dir, "model.pth")))
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs, les, les_wm = (
                    batch_data["mp2rage"].to(device),
                    batch_data["les"].type(torch.LongTensor).to(device),
                    #batch_data["tissue"].to(device),
                    batch_data["les_all"].type(torch.LongTensor).to(device))#.unsqueeze(0),)
                    
            roi_size = (108, 108, 108)
            sw_batch_size = 2 
    
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model,mode='gaussian')
            les = les.cpu().numpy()
            les_wm = les_wm.cpu().numpy()
            outputs2 = act(outputs).cpu().numpy()
            seg= np.squeeze(outputs2[:,1,:,:,:]+outputs2[:,2,:,:,:]*0.5)
            
            th = 0.5 #Set the threshold
            seg[seg>th]=1
            seg[seg<th]=0
            
            gt=les
            gt[gt>0]=1
            gt = np.squeeze(gt)    
            value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))

            metric_count += 1#len(value)
            metric_sum += value#.sum().item()
            print("Dice", metric_sum/metric_count)
            dice_all += metric_sum/metric_count
 
            ltpr_one, les_one = ltpr(seg,les,th)
            ltpr_all+=ltpr_one
            les_all+=les_one
            
            print("CLs",les_one)
            print("CLs detected",ltpr_one)
            lfpr_one, les_one_fpr = lfpr(seg,les,les_wm, th)
            lfpr_all+=lfpr_one
            les_all_fpr+=les_one_fpr
            print("False positives",lfpr_one)
            
            name_patient= os.path.basename(os.path.dirname(val_files[subject]["mp2rage"]))
            subject+=1
            meta_data = batch_data['mp2rage_meta_dict']
            for i, data in enumerate(outputs):  # save a batch of files
                out_meta = {k: meta_data[k][i] for k in meta_data} if meta_data else None
                 
            original_affine = out_meta.get("original_affine", None) if out_meta else None
            affine = out_meta.get("affine", None) if out_meta else None
            spatial_shape = out_meta.get("spatial_shape", None) if out_meta else None
            name = create_file_basename("subject_"+str(name_patient)+".nii.gz","probability_map",root_dir_res)
            data = outputs2
            data = np.squeeze(data)
            data = np.moveaxis(data,0,-1)

            write_nifti(data,name,affine=affine,target_affine=original_affine,
              output_spatial_shape=spatial_shape)
            
        lfpr_rate=lfpr_all/(les_all_fpr+0.00001)    
        ltpr_rate=ltpr_all/(les_all+0.00001)    
        print("LFPR",lfpr_rate)
        print("LTPR",ltpr_rate)
        print('Dice', dice_all/subject)
    
#%%
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)