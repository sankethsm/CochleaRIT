import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from metrics.metrics import GeneralizedDiceLoss

#-------------------------------------------------------------------------------#

def trainModel(epoch, model, trainLoader, optimizer, device, debugFlag, trainF=None):
    """
    Training routine for the model
    """

    model.train()
    nProcessed = 0
    finalLoss = 0
    listIOU = []
    nTotal = len(trainLoader)

    for batch_idx, dataDict in tqdm(enumerate(trainLoader), total=nTotal):
        data = dataDict['scan']
        target = dataDict['mask']
        
        data, target = data.to(device), target.to(device)
       
        optimizer.zero_grad() 
        output = model(data)
        #print(output.shape)

        criterion = GeneralizedDiceLoss(softmax=True)
        loss = criterion(output, target)
        
        loss.backward()
                
        optimizer.step()

        nProcessed += len(data)
        
        #Jaccard Index
        GT = torch.argmax(target, dim=1).cpu().numpy().astype(np.int).reshape(-1)
        PD = torch.argmax(output, dim=1).cpu().numpy().astype(np.int).reshape(-1)
        meanIOU = jaccard_score(GT, PD, labels=[0,1,2,3], average=None)
        
        listIOU.append(meanIOU)
        finalLoss += loss.data
        
    finalIOU = np.mean(np.mean(np.stack(listIOU, axis=0), axis=0),axis=0)
    finalLoss = finalLoss.cpu().numpy()
    print('Train Epoch: {} \tTrainDiceLoss: {:.8f}\tMeanTrainIOU: {:.8f}'.format(
            epoch, finalLoss/nTotal, finalIOU))
    
    nTotal = len(trainLoader)
    if not debugFlag:
        trainF.write('{},{},{}\n'.format(epoch, finalLoss/nTotal, listIOU))
        trainF.flush()
    
    return finalIOU, finalLoss/nTotal