"""
Written by Sanketh S. Moudgalya

Run script
"""
import os
import datetime
import time
import argparse
import sys
sys.path.append("..")

import torch
import torch.cuda
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter

from models.vnet_hl_new import VNet3D_HL
from models.vnet_hl_lowres import VNet3D_LR
from models.vnet_hl_dsus import VNet3D_DSUS_LR
from models.vnet_hl_parallel import VNet_Parll
from models.vnet_hl_dilated import VNet3D_Parallel_LR
from utils_.utils import adjust_opt, datestr, save_checkpoint
from datagen.dataGenerator import CochleaRIT, Augmentations, ToTensor
from scripts.train import trainModel
from scripts.valid import validModel
from scripts.arg_parser import argParser

#-------------------------------------------------------------------------------#

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = "cuda"
device = torch.device(dev)
#device = torch.device("cpu")
start_time = datetime.datetime.now()

#-------------------------------------------------------------------------------#
rootFolder = "C:\\Users\\sm5797\\Documents\\CochleaRIT\\"

#-------------------------------------------------------------------------------#
## Fill these first
modelTypeFlag = 'HR' # Takes HR, DSUS, ParaVNET, DilVNET
#sceFlag = 'HR' # Takes HR, LR and None
debugFlag = False # Set to True if you do not want logs to be created during debug
optims = ['adam']
lrs = [0.00001]#np.linspace(0.001, 0.00001, 10)
bsze = [1]
mm = 0.9
notes = modelTypeFlag +" VNet, Inital run."

#-------------------------------------------------------------------------------#

def main():
    parser = argParser()
    args = parser.parse_args()

    bestPred = 50.
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.cuda)

#-------------------------------------------------------------------------------#
    train = trainModel
    valid = validModel

#-------------------------------------------------------------------------------#  
    # Initialize optimizer and write out results and checkpoints
    for opts in optims:
        for l in lrs:
            for bs in bsze:
                torch.manual_seed(args.seed)
                if args.cuda:
                    torch.cuda.manual_seed(args.seed)
                
                if l==0.000001:
                    weight_decay = args.wd*0.1
                else:
                    weight_decay = args.wd
                print("build vnet")
                
                actvnType = 'prelu'
                dropRate = 0.05
                # Low resolution VNET. One part removed
                if modelTypeFlag == 'LR':
                    model = VNet3D_LR(reluType=actvnType, doRate=dropRate, seed=16)
                
                # Regular VNET. 
                elif modelTypeFlag == 'HR':
                    model = VNet3D_HL(reluType=actvnType, doRate=dropRate, seed=16)
                
                # Downsampling and upsampling of image before giving to a low resolution VNET
                elif modelTypeFlag == 'DSUS':
                    model = VNet3D_DSUS_LR(reluType=actvnType, doRate=dropRate, seed=16)
                
                # Parallel VNET with 3 branches having different dilations.
                elif modelTypeFlag == 'ParaVNET':
                    model = VNet_Parll(reluType=actvnType, doRate=dropRate, seed=16)
                
                # VNET with each convolutional loop in every part having different dilations.
                elif modelTypeFlag == 'DilVNET':
                    model = VNet3D_Parallel_LR(reluType=actvnType, doRate=dropRate, seed=16)
                
                if args.resume:
                    if os.path.isfile(args.resume):
                        print("=> loading checkpoint '{}'".format(args.resume))
                        checkpoint = torch.load(args.resume)
                        args.start_epoch = checkpoint['epoch']
                        bestPred = checkpoint['best_pred']
                        model.load_state_dict(checkpoint['state_dict'])
                        print("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.evaluate, checkpoint['epoch']))
                    else:
                        print("=> no checkpoint found at '{}'".format(args.resume))

                print('  + Number of params: {}'.format(
                        sum([p.data.nelement() for p in model.parameters()])))

                if dev=="cuda":
                    print('Using cuda')
                    model = model.cuda()
                elif dev=="cpu":
                    print('Using cpu')
                    model = model.cpu()
                
                #-------------------------------------------------------------------#
                if not debugFlag:
                    writer = SummaryWriter(logdir='.\\logs\\vnet_{}_{}_{}_{}'.format(datestr(), opts, bs, str(l)))
                    print(opts, l, bs)

                #-------------------------------------------------------------------#
                    sav_fol = args.save or '.\\torchRuns\\vnet_{}_{}_{}_{}'.format(datestr(), opts, bs, str(l))

                    if os.path.exists(sav_fol)==False:
                        os.mkdir(sav_fol)
                #-------------------------------------------------------------------#

                print("loading training set")
                trainSet = CochleaRIT('train', rootFolder, 
                transform=transforms.Compose([Augmentations('train', True), ToTensor()]))
                
                trainLoader = DataLoader(trainSet, batch_size=bs, shuffle=True)

                print("loading valid set")
    
                validSet = CochleaRIT('valid', rootFolder, 
                transform=transforms.Compose([Augmentations('valid', False), ToTensor()]))
    
                validSampler = RandomSampler(validSet, replacement=True, num_samples=5*len(validSet))

                validLoader = DataLoader(validSet, batch_size=1, shuffle=False, sampler=validSampler)
                
                if not debugFlag:
                    with open(os.path.join(sav_fol, "hyperparams_.csv"), 'w+') as wfil:
                        wfil.write("Train VNet for CochleaRIT dataset\n")
                        wfil.write("optimizer," + str(opts) + "\n")
                        wfil.write("loss func," + "Dice" + '\n')
                        wfil.write("learning rate," + str(l) + '\n')
                        wfil.write("train batch size," + str(bs) + '\n')
                        wfil.write("validation batch size," + str(1) + '\n')
                        wfil.write("momentum if SGD," + str(mm) + '\n')
                        wfil.write("total epochs," + str(args.nEpochs) + '\n')
                        wfil.write("augmentation type," + 
                        str('Elastic Deformations') + '\n')
                        wfil.write("augmentation alpha range," + str(25) + '-' + str(75) + '\n')
                        wfil.write("Activation type, " + actvnType + "\n")
                        wfil.write("Weight decay, " + str(weight_decay) + '\n')
                        wfil.write("Dropout, " + str(dropRate) + '\n')
                        wfil.write("number of convolutions (seed)," + str(16)+ '\n')
                        wfil.write("start time," + str(start_time) + '\n')
                        wfil.write("dataset," + 'CochleaRIT' + '\n')
                        #wfil.write("Regularization rate," + str(reguRate) + '\n')
                        wfil.write("Model,"+str(modelTypeFlag)+'\n')
                        wfil.write("Notes: " + notes + '\n')
                    trainF = open(os.path.join(sav_fol, 'train.csv'), 'w')
                    validF = open(os.path.join(sav_fol, 'valid.csv'), 'w')
                else:
                    trainF = None
                    validF = None

                #-------------------------------------------------------------------#
                if opts == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=l, momentum=mm,
                    weight_decay=weight_decay, nesterov=True)
                elif opts == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=l, 
                    weight_decay=weight_decay, amsgrad=True)
                
                for epoch in range(0, args.nEpochs + 1):
                    # Uncomment if changing learning rate is needed. Change LR values in subroutine
                    #adjust_opt(opts, optimizer, epoch, l)
                    
                    t0 = time.time()
                    
                    accuTrainIOU, lossTrain = train(epoch, model, trainLoader, optimizer, device, debugFlag, trainF)
                    
                    accuValidIOU, lossValid = valid(epoch, model, validLoader, device, debugFlag, validF)

                    print('time_elapsed {} seconds'.format(time.time()-t0)) 

                    if not debugFlag:
                        writer.add_scalars('Accuracy', {'train_dice':accuTrainIOU,
                                                'valid_dice':accuValidIOU}, global_step=epoch)

                    if epoch%10 == 0:
                        for name, param in model.named_parameters():
                            if not debugFlag:
                                writer.add_histogram(name, param, global_step=epoch)

                    if not debugFlag:
                        writer.add_scalars('Loss', {'train_loss':lossTrain,
                                                'valid_loss': lossValid}, global_step=epoch)

                    is_best = False
                    if (accuValidIOU*100) > bestPred:
                        is_best = True
                        bestPred = accuValidIOU*100
                    
                    save_checkpoint({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'bestPred': bestPred},
                                    is_best, sav_fol, "vnet")
                
                if not debugFlag:
                    trainF.close()
                    validF.close()

                    writer.close()

#-------------------------------------------------------------------------------#    

if __name__ == '__main__':
    
    main()