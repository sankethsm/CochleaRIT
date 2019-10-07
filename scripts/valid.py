import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_score

from ..metrics.metrics import GeneralizedDiceLoss

#-------------------------------------------------------------------------------#

def validModel(epoch, model, validLoader, device, debugFlag, validF=None):
    """
    Validation routine for the model
    """
    with torch.no_grad():
        model.eval()
        validLoss = 0
        listIOU = []
        totL1 = 0
        totL2 = 0
        nTotal = len(validLoader)

        for dict_ in validLoader:
            data = dict_['image']
            target = dict_['mask']
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            criterion = GeneralizedDiceLoss(softmax=True)
            loss = criterion(output, target)
            validLoss += loss.data

            #Jaccard Index
            GT = torch.argmax(target, dim=1).cpu().numpy().astype(np.int).reshape(-1)
            PD = torch.argmax(output, dim=1).cpu().numpy().astype(np.int).reshape(-1)
            meanIOU = jaccard_score(GT, PD, labels=[0,1,2,3], average=None)

            listIOU.append(meanIOU)
        
    finalIOU = np.mean(np.stack(listIOU, axis=0), axis=0)
    print('Valid Epoch: {} \tValidDiceLoss: {:.8f}\tMean IOU: {:.8f}'.format(
            epoch, validLoss/nTotal, finalIOU))

    if not debugFlag:
        validF.write('{},{},{}\n'.format(epoch, validLoss, finalIOU))
        validF.flush()

    return finalIOU, validLoss/nTotal