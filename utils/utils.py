import os
import shutil
import numpy as np
import time
import cv2
import torch


#-------------------------------------------------------------------------------#
def combineMasks(ST, SM, SV):
    """
    Combine three segmentation masks
    """
    STmask = np.where(ST>10., 1., 0.)
    SMmask = np.where(SM>10., 2., 0.)
    SVmask = np.where(SV>10., 3., 0.)
    
    combineTemp = STmask + SMmask
    combineTemp = np.where(combineTemp==3, 1, combineTemp)
    
    finMask = combineTemp + SVmask
    finMask = np.where(finMask==4, 1, finMask)
    finMask = np.where(finMask==5, 2, finMask)
    
    return finMask

def rszForModel(vol):
    rszScan = []
    finScan = []
    t1 = time.time()
    for i in range(vol.shape[2]):
        slc = vol[:,:,i]
        rszSlc = cv2.resize(slc, (256,256))
        rszScan.append(rszSlc)
    rszScan = np.array(rszScan)
        
    for j in range(rszScan.shape[2]):
        slc2 = rszScan[:,:,j]
        rszSlc2 = cv2.resize(slc2, (256,256))
        finScan.append(rszSlc2)
    finScan = np.array(finScan)
    t2 = time.time() - t1
    print('time taken to do interpolation: ',t2)
    
    '''
    volSz = vol.shape
    scale = tuple([256/x for x in volSz])
    t1 = time.time()
    rszVol = zoom(vol, zoom=scale)
    t2 = time.time() - t1
    print('time taken to do interpolation: ',t2)
    '''
    
    return finScan.transpose((2,0,1)) #rszVol

#-----------------------------------------------------------------------------------#
def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, 
                                    now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

#-----------------------------------------------------------------------------------#
def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    if os.path.exists(os.path.join(path, 'checkpoints'))==False:
        os.mkdir(os.path.join(path, 'checkpoints'))

    prefix_save = os.path.join(path, 'checkpoints', prefix)
    #print(prefix_save)
    epochNum = state['epoch']
    name = prefix_save + '_' + str(epochNum) + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

#-----------------------------------------------------------------------------------#
def noop(x):
    return x

#-----------------------------------------------------------------------------------#
def adjust_opt(optAlg, optimizer, epoch, lr):
    if optAlg == 'adam':
        if epoch < 40:
            lr = lr
        elif epoch == 40:
            lr = lr*1e-1
        elif epoch == 80:
            lr = lr*1e-1

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(param_group['lr'])

#-----------------------------------------------------------------------------------#