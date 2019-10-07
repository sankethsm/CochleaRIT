"""
Data generator for normal training
"""

import os
import glob
import random
import torch
import numpy as np
import pickle as pk
import cv2
import SimpleITK as sitk
import nibabel as nb
import time
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from albumentations import ElasticTransform

from ..utils.utils import combineMasks, rszForModel

#-------------------------------------------------------------------------------
class CochleaRIT(Dataset):

    def __init__(self, dataPartition, dataPath, transform=None):
    #def __init__(self, dataPath, transform=None):
        self.dataPath = dataPath
        self.dataList = []
        for foldr in os.listdir(self.dataPath):
            self.dataList.append(os.path.join(self.dataPath, foldr))
        self.dataPartition = dataPartition
        self.transform = transform

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, img_id):
        #imgID = self.dataList[img_id]
        imagePath = os.path.join(self.dataList[img_id], 'ear.nii')
        STPath = os.path.join(self.dataList[img_id], 'ST_labelmap.nii')
        SMPath = os.path.join(self.dataList[img_id], 'SM_labelmap.nii')
        SVPath = os.path.join(self.dataList[img_id], 'SV_labelmap.nii')
        
        img = nb.load(imagePath)
        scan = img.get_fdata()
        scan = rszForModel(scan)
        
        lm1 = nb.load(STPath)
        lm2 = nb.load(SMPath)
        lm3 = nb.load(SVPath)
        
        ST_temp = lm1.get_fdata()
        SM_temp = lm2.get_fdata()
        SV_temp = lm3.get_fdata()
        
        ST_temp = rszForModel(ST_temp)
        SM_temp = rszForModel(SM_temp)
        SV_temp = rszForModel(SV_temp)
        
        finalMask = combineMasks(ST_temp, SM_temp, SV_temp).astype(np.uint8)
        scan = np.divide(scan, 255.)

        dataDict = {'scan': scan, 'mask': finalMask}

        if self.transform:
            dataDict = self.transform(dataDict)

        return dataDict

#-------------------------------------------------------------------------------#
class Augmentations(object):
    def __init__(self, dataPartition, alpha, enable=None):
        self.dataPartition = dataPartition
        self.alpha = alpha
        self.enable = enable
        
    def __call__(self, dataDict):
        if self.enable:
            scan_, mask_ = dataDict['scan'], dataDict['mask']
            tx = ElasticTransform(p=1, alpha=self.alpha, sigma=self.alpha * 0.05, alpha_affine=self.alpha * 0.03)

            txdDataSample = tx(image=scan_, mask=mask_)

            elasticScan = txdDataSample['scan']
            elasticMask = txdDataSample['mask']

            elasticScan = elasticScan[np.newaxis, ...]
            elasticMask = elasticMask[np.newaxis, ...]
            dataDict = {'scan': elasticScan, 'mask': elasticMask}

        else:
            scan_, mask_ = dataDict['scan'], dataDict['mask']
            
            noTxScan = scan_[np.newaxis, ...]
            noTxMask = mask_[np.newaxis, ...]
            dataDict = {'scan': noTxScan, 'mask': noTxMask}

        return dataDict

#-------------------------------------------------------------------------------#
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        scan_, mask_ = sample['scan'], sample['mask']

        # swap color axis because
        # numpy image: H x W x D x C
        # torch image: C x H x W x D
        scan_ = scan_.transpose((0, 3, 1, 2))
        mask_ = mask_.transpose((0, 3, 1, 2))
        tempImg = torch.from_numpy(scan_)
        tempMsk = torch.from_numpy(mask_)

        tempImg = tempImg.type(torch.FloatTensor)
        tempMsk = tempMsk.type(torch.FloatTensor)
        return {'scan': tempImg, 'mask': tempMsk}
#-------------------------------------------------------------------------------#
