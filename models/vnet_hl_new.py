## Base VNET model used at Health Lab

import torch
import torch.nn as nn

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
        #self.bnorm = nn.InstanceNorm3d(nchan)

    def forward(self, x):
        #self.__init__(input)
        return self.bnorm(x)

class Dropout3D_(nn.Module):
    def __init__(self,doRate):
        super(Dropout3D_, self).__init__()
        self.do = nn.Dropout3d(doRate)

    def forward(self, x):
        return self.do(x)

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, reluType, doRate):
        super(DownTransition, self).__init__()
        self.nConvs = nConvs
        self.downInputConv = nn.Conv3d(inChans, outChans, kernel_size=5, stride=1, padding=2)
        self.downInBnorm = BatchNorm3D_(outChans)
        self.downInActvn = activationChoice(reluType)
        self.downConv = nn.Conv3d(outChans, outChans, kernel_size=2, stride=2, padding=0)
        self.downConvLoop = nn.ModuleList([nn.Conv3d(outChans, outChans, kernel_size=5,
                                        padding=2) for i in range(self.nConvs)])
        self.downBnormLoop = nn.ModuleList([BatchNorm3D_(outChans) for i in range(self.nConvs)])
        self.downActvnLoop = nn.ModuleList([activationChoice(reluType) for i in range(self.nConvs)])
        self.downOutBnorm = BatchNorm3D_(outChans)
        self.downOutDo = Dropout3D_(doRate)
        self.downOutActvn = activationChoice(reluType)

    def forward(self, x, needBnorm=True, doNeed=True):
        
        inp = self.downInputConv(x)
        if needBnorm==True:
            inp = self.downInBnorm(inp)
        inp = self.downInActvn(inp)

        for i in range(self.nConvs):
            inp = self.downConvLoop[i](inp)
            if needBnorm==True:
                inp = self.downBnormLoop[i](inp)
            inp = self.downActvnLoop[i](inp)
        downInCat = torch.cat([x]*2, 1)

        addDown = torch.add(inp, downInCat)

        downsample = self.downConv(addDown)
        downsample = self.downOutActvn(downsample)
        
        if needBnorm==True:
            downsample = self.downOutBnorm(downsample)
        if doNeed==True:
            downsample = self.downOutDo(downsample)
        
        return downsample, addDown

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, reluType, doRate):
        super(UpTransition, self).__init__()
        self.nConvs = nConvs
        self.upInputConv = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.upInBnorm = BatchNorm3D_(outChans)
        self.upInActvn = activationChoice(reluType)
        self.upConvLoop = nn.ModuleList([nn.Conv3d(outChans, outChans, kernel_size=5, 
                                    padding=2) for i in range(self.nConvs)])
        self.upBnormLoop = nn.ModuleList([BatchNorm3D_(outChans) for i in range(self.nConvs)])
        self.upActvnLoop = nn.ModuleList([activationChoice(reluType) for i in range(self.nConvs)])
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

        for i in range(self.nConvs):
            inp = self.upConvLoop[i](inp)
            inp = self.upBnormLoop[i](inp)
            inp = self.upActvnLoop[i](inp)

        addUp = torch.add(inp, x0)
        upsample = self.upSample(addUp)
        upsample = self.upOutActvn(upsample)
        
        if needBnorm==True:
            upsample = self.upOutBnorm(upsample)
        if doNeed==True:
            upsample = self.upOutDo(upsample)
        
        return upsample


class VNet3D_HL(nn.Module):
    def __init__(self, reluType, doRate, seed=16, convs=3):
        super(VNet3D_HL, self).__init__()
        self.seed = seed
        self.convIn1 = nn.Conv3d(1, seed, kernel_size=5, padding=2)
        self.bNormIn1 = BatchNorm3D_(seed)
        self.actvnIn1 = activationChoice(reluType)

        self.ds1 = nn.Conv3d(seed, seed, kernel_size=2, stride=2, padding=0)
        self.bnormDS1 = BatchNorm3D_(seed)
        self.doDS1 = Dropout3D_(doRate)
        self.actvnDS1 = activationChoice(reluType)
        
        self.downTr2 = DownTransition(seed, 2*seed, 2, reluType, doRate)
        self.downTr3 = DownTransition(2*seed, 4*seed, convs, reluType, doRate)
        self.downTr4 = DownTransition(4*seed, 8*seed, convs, reluType, doRate)

        self.conv51 = nn.Conv3d(8*seed, 16*seed, kernel_size=5, padding=2)
        self.bnorm51 = BatchNorm3D_(16*seed)
        self.actvn51 = activationChoice(reluType)

        self.conv52 = nn.Conv3d(16*seed, 16*seed, kernel_size=5, padding=2)
        self.bnorm52 = BatchNorm3D_(16*seed)
        self.actvn52 = activationChoice(reluType)

        self.conv53 = nn.Conv3d(16*seed, 16*seed, kernel_size=5, padding=2)
        self.bnorm53 = BatchNorm3D_(16*seed)
        self.actvn53 = activationChoice(reluType)

        self.us4 = nn.ConvTranspose3d(16*seed, 8*seed, kernel_size=2, stride=2, padding=0)
        self.bnormUS4 = BatchNorm3D_(8*seed)
        self.doUS4 = Dropout3D_(doRate)
        self.actvnUS4 = activationChoice(reluType)

        self.upTr3 = UpTransition(16*seed, 8*seed, convs, reluType, doRate)
        self.upTr2 = UpTransition(8*seed, 4*seed, convs, reluType, doRate)
        self.upTr1 = UpTransition(4*seed, 2*seed, 2, reluType, doRate)

        self.convOut1 = nn.Conv3d(2*seed, seed, kernel_size=5, padding=2)
        self.bNormOut1 = BatchNorm3D_(seed)
        self.actvnOut1 = activationChoice(reluType)

        self.convOutFinal = nn.Conv3d(seed, 4, kernel_size=1, stride=1, padding=0)

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

        out52 = self.conv52(out51)
        if bnormNeed==True:
            out52 = self.bnorm52(out52)
        out52 = self.actvn52(out52)

        out53 = self.conv52(out52)
        if bnormNeed==True:
            out53 = self.bnorm53(out53)
        out53 = self.actvn53(out53)

        out4Cat = torch.cat([out4ds]*2, 1)

        out5Add = torch.add(out53, out4Cat)

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

        #outSigmoid = self.sigmoid(outTrLast)

        return outTrLast#outSigmoid








