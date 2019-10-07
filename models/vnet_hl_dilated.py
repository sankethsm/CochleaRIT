## VNET with dilations in conv loop part

import torch
import torch.nn as nn
import torch.nn.functional as F

def activationChoice(reluType):
    if reluType=='leakyrelu':
        return nn.LeakyReLU(0.1)
    elif reluType=='prelu':
        return nn.PReLU()
    elif reluType=='relu':
        return nn.ReLU()
    else:
        return nn.ReLU()

class BatchNorm3D_(nn.Module):
    def __init__(self, nchan):
        super(BatchNorm3D_, self).__init__()
        self.bnorm = nn.BatchNorm3d(nchan)
        ## uncomment if you need instance normalization instead of batchnorm in the network
        #self.bnorm = nn.InstanceNorm3d(nchan)

    def forward(self, x):
        return self.bnorm(x)

class Dropout3D_(nn.Module):
    def __init__(self,doRate):
        super(Dropout3D_, self).__init__()
        self.do = nn.Dropout3d(doRate)

    def forward(self, x):
        return self.do(x)

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, reluType, doRate, nConvs):
        super(DownTransition, self).__init__()
        self.nConvs = nConvs
        self.downInputConv = nn.Conv3d(inChans, outChans, kernel_size=5, stride=1, padding=2)
        self.downInBnorm = BatchNorm3D_(outChans)
        self.downInActvn = activationChoice(reluType)
        
        if self.nConvs == 3:
            # with dilation 1
            self.downConvLoop1 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,1,1), dilation=(1,1,1))
            self.downBnormLoop1 = BatchNorm3D_(outChans)
            self.downActvnLoop1 = activationChoice(reluType)

            # with dilation 2
            self.downConvLoop2 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,2,2), dilation=(1,2,2))
            self.downBnormLoop2 = BatchNorm3D_(outChans)
            self.downActvnLoop2 = activationChoice(reluType)

            # with dilation 3
            self.downConvLoop3 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,3,3), dilation=(1,3,3))
            self.downBnormLoop3 = BatchNorm3D_(outChans)
            self.downActvnLoop3 = activationChoice(reluType)

            # Concatinating all three dilation convolutions
            self.downLoopCat = nn.Conv3d(3*outChans, outChans, kernel_size=3, padding=1)
            self.downLoopCatNorm = BatchNorm3D_(outChans)
            self.downLoopCatActvn = activationChoice(reluType)
        
        elif self.nConvs == 2:
            # with dilation 1
            self.downConvLoop1 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,1,1), dilation=(1,1,1))
            self.downBnormLoop1 = BatchNorm3D_(outChans)
            self.downActvnLoop1 = activationChoice(reluType)

            # with dilation 2
            self.downConvLoop2 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,2,2), dilation=(1,2,2))
            self.downBnormLoop2 = BatchNorm3D_(outChans)
            self.downActvnLoop2 = activationChoice(reluType)

            # Concatinating two dilation convolutions
            self.downLoopCat = nn.Conv3d(2*outChans, outChans, kernel_size=3, padding=1)
            self.downLoopCatNorm = BatchNorm3D_(outChans)
            self.downLoopCatActvn = activationChoice(reluType)


        self.downConv = nn.Conv3d(outChans, outChans, kernel_size=2, stride=2, padding=0)
        self.downOutBnorm = BatchNorm3D_(outChans)
        self.downOutDo = Dropout3D_(doRate)
        self.downOutActvn = activationChoice(reluType)

    def forward(self, x, needBnorm=True, doNeed=True):
        
        inp = self.downInputConv(x)
        if needBnorm==True:
            inp = self.downInBnorm(inp)
        inp = self.downInActvn(inp)
        
        if self.nConvs == 3:
            inp1 = self.downConvLoop1(inp)
            if needBnorm==True:
                inp1 = self.downBnormLoop1(inp1)
            inp1 = self.downActvnLoop1(inp1)

            inp2 = self.downConvLoop2(inp)
            if needBnorm==True:
                inp2 = self.downBnormLoop2(inp2)
            inp2 = self.downActvnLoop2(inp2)

            inp3 = self.downConvLoop3(inp)
            if needBnorm==True:
                inp3 = self.downBnormLoop3(inp3)
            inp3 = self.downActvnLoop3(inp3)

            combineLoopDown = torch.cat([inp1, inp2, inp3], 1)

            catLoopDown = self.downLoopCat(combineLoopDown)
            catLoopDown = self.downLoopCatNorm(catLoopDown)
            catLoopDown = self.downLoopCatActvn(catLoopDown)
        
        elif self.nConvs == 2:
            inp1 = self.downConvLoop1(inp)
            if needBnorm==True:
                inp1 = self.downBnormLoop1(inp1)
            inp1 = self.downActvnLoop1(inp1)

            inp2 = self.downConvLoop2(inp)
            if needBnorm==True:
                inp2 = self.downBnormLoop2(inp2)
            inp2 = self.downActvnLoop2(inp2)

            combineLoopDown = torch.cat([inp1, inp2], 1)

            catLoopDown = self.downLoopCat(combineLoopDown)
            catLoopDown = self.downLoopCatNorm(catLoopDown)
            catLoopDown = self.downLoopCatActvn(catLoopDown)

        downInCat = torch.cat([x]*2, 1)

        addDown = torch.add(catLoopDown, downInCat)

        downsample = self.downConv(addDown)
        downsample = self.downOutActvn(downsample)
        
        if needBnorm==True:
            downsample = self.downOutBnorm(downsample)
        if doNeed==True:
            downsample = self.downOutDo(downsample)
        
        return downsample, addDown

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, reluType, doRate, nConvs):
        super(UpTransition, self).__init__()
        self.nConvs = nConvs
        self.upInputConv = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.upInBnorm = BatchNorm3D_(outChans)
        self.upInActvn = activationChoice(reluType)
        
        if self.nConvs == 3:
            # with dilation 1
            self.upConvLoop1 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,1,1), dilation=(1,1,1))
            self.upBnormLoop1 = BatchNorm3D_(outChans)
            self.upActvnLoop1 = activationChoice(reluType)

            # with dilation 2
            self.upConvLoop2 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,2,2), dilation=(1,2,2))
            self.upBnormLoop2 = BatchNorm3D_(outChans)
            self.upActvnLoop2 = activationChoice(reluType)

            # with dilation 3
            self.upConvLoop3 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,3,3), dilation=(1,3,3))
            self.upBnormLoop3 = BatchNorm3D_(outChans)
            self.upActvnLoop3 = activationChoice(reluType)

            # Concatinating all three dilation convolutions
            self.upLoopCat = nn.Conv3d(3*outChans, outChans, kernel_size=3, padding=1)
            self.upLoopCatNorm = BatchNorm3D_(outChans)
            self.upLoopCatActvn = activationChoice(reluType)
        
        elif self.nConvs == 2:
            # with dilation 1
            self.upConvLoop1 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,1,1), dilation=(1,1,1))
            self.upBnormLoop1 = BatchNorm3D_(outChans)
            self.upActvnLoop1 = activationChoice(reluType)

            # with dilation 2
            self.upConvLoop2 = nn.Conv3d(outChans, outChans, kernel_size=3, 
                                    padding=(1,2,2), dilation=(1,2,2))
            self.upBnormLoop2 = BatchNorm3D_(outChans)
            self.upActvnLoop2 = activationChoice(reluType)

            # Concatinating two dilation convolutions
            self.upLoopCat = nn.Conv3d(2*outChans, outChans, kernel_size=3, padding=1)
            self.upLoopCatNorm = BatchNorm3D_(outChans)
            self.upLoopCatActvn = activationChoice(reluType)
        

        self.upSample = nn.ConvTranspose3d(outChans, outChans // 2, 
                                            kernel_size=2, stride=2, padding=0)
        self.upOutBnorm = BatchNorm3D_(outChans // 2)
        self.upOutDo = Dropout3D_(doRate)
        self.upOutActvn = activationChoice(reluType)

    def forward(self, x0, x1, needBnorm=True, doNeed=True):
        inMerge = torch.cat([x0, x1], 1)
        inp = self.upInputConv(inMerge)
        if needBnorm==True:
            inp = self.upInBnorm(inp)
        inp = self.upInActvn(inp)

        if self.nConvs == 3:
            inp1 = self.upConvLoop1(inp)
            if needBnorm==True:
                inp1 = self.upBnormLoop1(inp1)
            inp1 = self.upActvnLoop1(inp1)

            inp2 = self.upConvLoop2(inp)
            if needBnorm==True:
                inp2 = self.upBnormLoop2(inp2)
            inp2 = self.upActvnLoop2(inp2)

            inp3 = self.upConvLoop3(inp)
            if needBnorm==True:
                inp3 = self.upBnormLoop3(inp3)
            inp3 = self.upActvnLoop3(inp3)

            combineLoopUp = torch.cat([inp1, inp2, inp3], 1)

            catLoopUp = self.upLoopCat(combineLoopUp)
            catLoopUp = self.upLoopCatNorm(catLoopUp)
            catLoopUp = self.upLoopCatActvn(catLoopUp)
        
        elif self.nConvs == 2:
            inp1 = self.upConvLoop1(inp)
            if needBnorm==True:
                inp1 = self.upBnormLoop1(inp1)
            inp1 = self.upActvnLoop1(inp1)

            inp2 = self.upConvLoop2(inp)
            if needBnorm==True:
                inp2 = self.upBnormLoop2(inp2)
            inp2 = self.upActvnLoop2(inp2)

            combineLoopUp = torch.cat([inp1, inp2], 1)

            catLoopUp = self.upLoopCat(combineLoopUp)
            catLoopUp = self.upLoopCatNorm(catLoopUp)
            catLoopUp = self.upLoopCatActvn(catLoopUp)

        addUp = torch.add(catLoopUp, x0)
        upsample = self.upSample(addUp)
        upsample = self.upOutActvn(upsample)
        
        if needBnorm==True:
            upsample = self.upOutBnorm(upsample)
        if doNeed==True:
            upsample = self.upOutDo(upsample)
        
        return upsample


class VNet3D_Parallel_HR(nn.Module):
    ### All part of VNET retained
    def __init__(self, reluType, doRate, seed=16):
        super(VNet3D_Parallel_HR, self).__init__()
        self.seed = seed
        self.convIn1 = nn.Conv3d(1, seed, kernel_size=5, padding=2)
        self.bNormIn1 = BatchNorm3D_(seed)
        self.actvnIn1 = activationChoice(reluType)

        self.ds1 = nn.Conv3d(seed, seed, kernel_size=2, stride=2, padding=0)
        self.bnormDS1 = BatchNorm3D_(seed)
        self.doDS1 = Dropout3D_(doRate)
        self.actvnDS1 = activationChoice(reluType)
        
        self.downTr2 = DownTransition(seed, 2*seed, reluType, doRate, 2)
        self.downTr3 = DownTransition(2*seed, 4*seed, reluType, doRate, 3)
        self.downTr4 = DownTransition(4*seed, 8*seed, reluType, doRate, 3)

        ##
        self.conv51 = nn.Conv3d(8*seed, 16*seed, kernel_size=3, padding=(1,1,1), dilation=(1,1,1))
        self.bnorm51 = BatchNorm3D_(16*seed)
        self.actvn51 = activationChoice(reluType)
        
        self.conv52 = nn.Conv3d(8*seed, 16*seed, kernel_size=3, padding=(1,2,2), dilation=(1,2,2))
        self.bnorm52 = BatchNorm3D_(16*seed)
        self.actvn52 = activationChoice(reluType)

        self.conv53 = nn.Conv3d(8*seed, 16*seed, kernel_size=3, padding=(1,3,3), dilation=(1,3,3))
        self.bnorm53 = BatchNorm3D_(16*seed)
        self.actvn53 = activationChoice(reluType)

        self.combine5 = nn.Conv3d(3*16*seed, 16*seed, kernel_size=3, padding=1)
        self.combineBnorm5 = BatchNorm3D_(16*seed)
        self.combineActvn5 = activationChoice(reluType)
        ##

        self.us4 = nn.ConvTranspose3d(16*seed, 8*seed, kernel_size=2, stride=2, padding=0)
        self.bnormUS4 = BatchNorm3D_(8*seed)
        self.doUS4 = Dropout3D_(doRate)
        self.actvnUS4 = activationChoice(reluType)

        self.upTr3 = UpTransition(16*seed, 8*seed, reluType, doRate, 2)
        self.upTr2 = UpTransition(8*seed, 4*seed, reluType, doRate, 3)
        self.upTr1 = UpTransition(4*seed, 2*seed, reluType, doRate, 3)

        self.convOut1 = nn.Conv3d(2*seed, seed, kernel_size=5, padding=2)
        self.bNormOut1 = BatchNorm3D_(seed)
        self.actvnOut1 = activationChoice(reluType)

        self.convOutFinal = nn.Conv3d(seed, 1, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, bnormNeed=True, doNeed=True):
        inTr = self.convIn1(x)
        if bnormNeed==True:
            inTr = self.bNormIn1(inTr)
        inActvn = self.actvnIn1(inTr)

        inCat = torch.cat([x]*self.seed, 1)
        inAdd = torch.add(inActvn, inCat)

        down1 = self.ds1(inAdd)
        down1 = self.actvnDS1(down1)
        
        if bnormNeed==True:
            down1 = self.bnormDS1(down1)
        if doNeed==True:
            down1 = self.doDS1(down1)
        out1ds = down1

        out2ds, skip2 = self.downTr2(out1ds)
        out3ds, skip3 = self.downTr3(out2ds)
        out4ds, skip4 = self.downTr4(out3ds)

        out51 = self.conv51(out4ds)
        if bnormNeed==True:
            out51 = self.bnorm51(out51)
        out51 = self.actvn51(out51)
        
        out52 = self.conv52(out4ds)
        if bnormNeed==True:
            out52 = self.bnorm52(out52)
        out52 = self.actvn52(out52)

        out53 = self.conv52(out4ds)
        if bnormNeed==True:
            out53 = self.bnorm53(out53)
        out53 = self.actvn53(out53)
        
        out5Cat = torch.cat([out51, out52, out53], 1)

        out5Combine = self.combine5(out5Cat)
        out5Combine = self.combineBnorm5(out5Combine)
        out5Combine = self.combineActvn5(out5Combine)

        out4Cat = torch.cat([out4ds]*2, 1)

        out5Add = torch.add(out5Combine, out4Cat)

        out4us = self.us4(out5Add)
        out4us = self.actvnUS4(out4us)
        
        if bnormNeed==True:
            out4us = self.bnormUS4(out4us)
        if doNeed==True:
            out4us = self.doUS4(out4us)

        out3us = self.upTr3(out4us, skip4)
        out2us = self.upTr2(out3us, skip3)
        out1us = self.upTr1(out2us, skip2)

        outCat = torch.cat([out1us, inAdd], 1)

        outTr = self.convOut1(outCat)
        if bnormNeed==True:
            outTr = self.bNormOut1(outTr)
        outTr = self.actvnOut1(outTr)

        outAdd = torch.add(outTr, out1us)
        outTrLast = self.convOutFinal(outAdd)

        outSigmoid = self.sigmoid(outTrLast)

        return outSigmoid


class VNet3D_Parallel_LR(nn.Module):
    ## One part of VNET removed.
    def __init__(self, reluType, doRate, seed=16):
        super(VNet3D_Parallel_LR, self).__init__()
        self.seed = seed
        self.convIn1 = nn.Conv3d(1, seed, kernel_size=5, padding=2)
        self.bNormIn1 = BatchNorm3D_(seed)
        self.actvnIn1 = activationChoice(reluType)

        self.ds1 = nn.Conv3d(seed, seed, kernel_size=2, stride=2, padding=0)
        self.bnormDS1 = BatchNorm3D_(seed)
        self.doDS1 = Dropout3D_(doRate)
        self.actvnDS1 = activationChoice(reluType)
        
        self.downTr2 = DownTransition(seed, 2*seed, reluType, doRate, 2)
        self.downTr3 = DownTransition(2*seed, 4*seed, reluType, doRate, 3)

        ##
        self.conv41 = nn.Conv3d(4*seed, 8*seed, kernel_size=3, padding=(1,1,1), dilation=(1,1,1))
        self.bnorm41 = BatchNorm3D_(8*seed)
        self.actvn41 = activationChoice(reluType)
        
        self.conv42 = nn.Conv3d(4*seed, 8*seed, kernel_size=3, padding=(1,2,2), dilation=(1,2,2))
        self.bnorm42 = BatchNorm3D_(8*seed)
        self.actvn42 = activationChoice(reluType)

        self.conv43 = nn.Conv3d(4*seed, 8*seed, kernel_size=3, padding=(1,3,3), dilation=(1,3,3))
        self.bnorm43 = BatchNorm3D_(8*seed)
        self.actvn43 = activationChoice(reluType)

        self.combine4 = nn.Conv3d(3*8*seed, 8*seed, kernel_size=3, padding=1)
        self.combineBnorm4 = BatchNorm3D_(8*seed)
        self.combineActvn4 = activationChoice(reluType)
        ##

        self.us3 = nn.ConvTranspose3d(8*seed, 4*seed, kernel_size=2, stride=2, padding=0)
        self.bnormUS3 = BatchNorm3D_(4*seed)
        self.doUS3 = Dropout3D_(doRate)
        self.actvnUS3 = activationChoice(reluType)

        self.upTr2 = UpTransition(8*seed, 4*seed, reluType, doRate, 2)
        self.upTr1 = UpTransition(4*seed, 2*seed, reluType, doRate, 3)

        self.convOut1 = nn.Conv3d(2*seed, seed, kernel_size=5, padding=2)
        self.bNormOut1 = BatchNorm3D_(seed)
        self.actvnOut1 = activationChoice(reluType)

        self.convOutFinal = nn.Conv3d(seed, 1, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        ## incase of low resolution SCE, use upsample
        self.usLR = nn.Upsample(size=(48, 128, 192), mode='trilinear')


    def forward(self, x, bnormNeed=True, doNeed=True):
        inDS = F.interpolate(x, size=(48,64,96), mode='trilinear')
        inTr = self.convIn1(inDS)
        if bnormNeed==True:
            inTr = self.bNormIn1(inTr)
        inActvn = self.actvnIn1(inTr)

        inCat = torch.cat([inDS]*self.seed, 1)
        inAdd = torch.add(inActvn, inCat)

        down1 = self.ds1(inAdd)
        down1 = self.actvnDS1(down1)
        
        if bnormNeed==True:
            down1 = self.bnormDS1(down1)
        if doNeed==True:
            down1 = self.doDS1(down1)
        out1ds = down1

        out2ds, skip2 = self.downTr2(out1ds)
        out3ds, skip3 = self.downTr3(out2ds)

        out41 = self.conv41(out3ds)
        if bnormNeed==True:
            out41 = self.bnorm41(out41)
        out41 = self.actvn41(out41)
        
        out42 = self.conv42(out3ds)
        if bnormNeed==True:
            out42 = self.bnorm42(out42)
        out42 = self.actvn42(out42)

        out43 = self.conv43(out3ds)
        if bnormNeed==True:
            out43 = self.bnorm43(out43)
        out43 = self.actvn43(out43)
        
        out4Cat = torch.cat([out41, out42, out43], 1)

        out4Combine = self.combine4(out4Cat)
        out4Combine = self.combineBnorm4(out4Combine)
        out4Combine = self.combineActvn4(out4Combine)

        out4Cat2 = torch.cat([out3ds]*2, 1)

        out4Add = torch.add(out4Combine, out4Cat2)

        out3us = self.us3(out4Add)
        out3us = self.actvnUS3(out3us)
        
        if bnormNeed==True:
            out3us = self.bnormUS3(out3us)
        if doNeed==True:
            out3us = self.doUS3(out3us)

        out2us = self.upTr2(out3us, skip3)
        out1us = self.upTr1(out2us, skip2)

        outCat = torch.cat([out1us, inAdd], 1)

        outTr = self.convOut1(outCat)
        if bnormNeed==True:
            outTr = self.bNormOut1(outTr)
        outTr = self.actvnOut1(outTr)

        outAdd = torch.add(outTr, out1us)
        outTrLast = self.convOutFinal(outAdd)

        outSigmoid = self.sigmoid(outTrLast)

        ## incase of low resolution SCE
        outSigmoid = self.usLR(outSigmoid)

        return outSigmoid
        







