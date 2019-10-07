import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_score

from ..metrics.metrics import GeneralizedDiceLoss

#-------------------------------------------------------------------------------#

def trainModel(epoch, model, trainLoader, optimizer, device, debugFlag, trainF=None):
    """
    Training routine for the model
    """

    model.train()
    nProcessed = 0
    finalLoss = 0
    listIOU = []
    nTrain = len(trainLoader.dataset)
    nTotal = len(trainLoader)

    for batch_idx, dataDict in enumerate(trainLoader):
        data = dataDict['scan']
        target = dataDict['mask']
        
        data, target = data.to(device), target.to(device)
       
        optimizer.zero_grad() 
        output = model(data)

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
        
    finalIOU = np.mean(np.stack(listIOU, axis=0), axis=0)
    print('Train Epoch: {} \tDiceLoss: {:.8f}\tMean IOU: {:.8f}'.format(
            epoch, finalLoss/nTotal, finalIOU))
    
    nTotal = len(trainLoader)
    if not debugFlag:
        trainF.write('{},{},{}\n'.format(epoch, finalLoss/nTotal, listIOU))
        trainF.flush()
    
    return finalIOU, finalLoss/nTotal