from collections import OrderedDict

import torch
import torch.nn as nn

import scipy.io as sio
import numpy as np

class BaseDenseLayer(nn.Module):
    """docstring for BaseLayer"""
    def __init__(self, input_dim, growth_rate):
        super(BaseDenseLayer, self).__init__()
           
        self.layer = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=growth_rate, kernel_size=3, padding=1)
            )        

    def forward(self, x):
        # input('BaseDenseLayer:forward loop ')
        out = self.layer(x)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, inputDim=256, outputDim=256, growthRate=32, blockDepth=8):
        super(DenseBlock, self).__init__()

        self.inputDim = inputDim
        self.outputDim = outputDim
        self.blockDepth = blockDepth
        self.growthRate = growthRate


        layers = []
        for indx in range(self.blockDepth):
            srcDim = self.inputDim + indx*self.growthRate
            layers.append(BaseDenseLayer(srcDim, self.growthRate))

        self.denseLayer = nn.Sequential(*layers)
        
        catDim = self.inputDim+self.blockDepth*self.growthRate
        self.transition = nn.Sequential(
            nn.BatchNorm2d(catDim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=catDim, out_channels=self.outputDim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # input('DenseBlock:forward')
        # catOut = x
        # for indx in range(self.blockDepth):
        #     input('DenseBlock:forward loop %d'%indx)
        #     currOut = self.layers[indx](catOut)
        #     catOut = torch.cat([catOut, currOut], 1)
        out = self.denseLayer(x)
        out = self.transition(out)
        # print(out)
        return out

    # def baseLayer(self, indx):
    #     srcDim = self.inputDim+indx*self.growthRate
    #     return list([
    #         nn.BatchNorm2d(srcDim),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(in_channels=srcDim, out_channels=128, kernel_size=1, bias=False),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(in_channels=128, out_channels=self.growthRate, kernel_size=3, padding=1)
    #     ])

class DFCN_32(nn.Module):
    """docstring for DFCN_32"""
    def __init__(self):
        super(DFCN_32, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.relu1 =  nn.ReLU(inplace=True)
        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.denseBlock1 = DenseBlock(inputDim=64, outputDim=128 ,growthRate=32, blockDepth=4)
        self.relu2 =  nn.ReLU(inplace=True)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.denseBlock2 = DenseBlock(inputDim=128, outputDim=256 ,growthRate=32, blockDepth=8)
        self.relu3 =  nn.ReLU(inplace=True)
        self.pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.denseBlock3 = DenseBlock(inputDim=256, outputDim=512 ,growthRate=32, blockDepth=16)
        self.relu4 =  nn.ReLU(inplace=True)
        self.pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_fc_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
        self.relu5 =  nn.ReLU(inplace=True)
        self.conv_fc_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
        self.relu6 =  nn.ReLU(inplace=True)

        self.score_32 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)
        self.relu7 =  nn.ReLU(inplace=True)
        self.upsample_to_16 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample_to_8 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample_to_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample4x = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu1(out)
        out = self.pooling_1(out)

        out = self.denseBlock1(out)
        out = self.relu2(out)
        out = self.pooling_2(out)
        out = self.denseBlock2(out)
        out = self.relu3(out)
        out = self.pooling_3(out)
        out = self.denseBlock3(out)
        out = self.relu4(out)
        out = self.pooling_4(out)

        out = self.conv_fc_5_1(out)
        out = self.relu5(out)
        out = self.conv_fc_5_2(out)
        out = self.relu6(out)
        out = self.score_32(out)
        out = self.relu7(out)

        out = self.upsample_to_16(out)
        out = self.upsample_to_8(out)
        out = self.upsample_to_4(out)
        out = self.upsample4x(out)

        return out


class RDCN_VGG(nn.Module):
    def __init__(self, rec_num):
        super(RDCN_VGG, self).__init__()

        self.recNum = rec_num
        self.downsample = nn.Sequential(OrderedDict([
            ('data/bn', nn.BatchNorm2d(3)),
            ('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
            ('conv1_1/bn', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('conv1_2/bn', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('conv2_1/bn', nn.BatchNorm2d(128)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ('conv2_2/bn', nn.BatchNorm2d(128)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_1/bn', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_2/bn', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_3/bn', nn.BatchNorm2d(256)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('conv3_4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_4/bn', nn.BatchNorm2d(256)),
            ('relu3_4', nn.ReLU(inplace=True))
        ]))

        self.denseBlock = DenseBlock(inputDim=256, outputDim=256 ,growthRate=32, blockDepth=8)
        self.predictx4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        self.weightedAvg = nn.Conv2d(in_channels=self.recNum, out_channels=1, kernel_size=1, bias=True)

        self.upsample4x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )  
        self.predict = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False)

    def loadConv(self, pretrain_model):
        pretrainModel = sio.loadmat(pretrain_model)
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name.split('.')[-1]
                if module.bias is not None:
                    for key, value in pretrainModel.items():
                        if '%s_0'%last_name == key:  # for weight
                            print('load %s'%key)
                            self.copyArrayToTensor(value, module.weight.data)

                        if '%s_1'%last_name == key:  # for weight
                            print('load %s'%key)
                            self.copyArrayToTensor(value, module.bias.data)
                else:
                    for key, value in pretrainModel.items():
                        if '%s_0'%last_name == key:  # for weight
                            print('load %s'%key)
                            self.copyArrayToTensor(value, module.weight.data)


    def copyArrayToTensor(self, array, tensor):
        aShape = array.shape
        tShape = tensor.shape
        
        if len(aShape) == 2 and aShape[0] == 1:
            array = np.squeeze(array)
            aShape = array.shape

        if len(aShape) != len(tShape):
            raise ValueError('array shape:{} mismatches with tensor: {}'.format(aShape, tShape))

        for indx in range(len(aShape)):
            if aShape[indx] != tShape[indx]:
                raise ValueError('array shape:{} mismatches with tensor: {}'.format(aShape, tShape))

        if len(aShape) == 1:
            for n in range(aShape[0]):
                tensor[n] = float(array[n])
        elif len(aShape) == 2:
            for n in range(aShape[0]):
                for c in range(aShape[1]):
                    tensor[n, c] = float(array[n, c])
        elif len(aShape) == 3:
            for n in range(aShape[0]):
                for c in range(aShape[1]):
                    for h in range(aShape[2]):
                        tensor[n, c, h] = float(array[n, c, h])
        elif len(aShape) == 4:
            for n in range(aShape[0]):
                for c in range(aShape[1]):
                    for h in range(aShape[2]):
                        for w in range(aShape[3]):
                            tensor[n, c, h, w] = float(array[n, c, h, w])


    def forward(self, x):
        out = self.downsample(x)
        predictx4s = [None for i in range(self.recNum)]
        catFlag = False
        predictx4Cat = None
        predict_final = None

        # input("RDCN_VGG before loop")
        for indx in range(self.recNum):
            out = self.denseBlock(out)
            predictx4s[indx] = self.predictx4(out)
            if not catFlag:
                catFlag = True
                predictx4Cat = predictx4s[indx]
            else:
                predictx4Cat = torch.cat([predictx4Cat, predictx4s[indx]], 1)
            # print(predictx4s[indx])

        predictx4_avg = self.weightedAvg(predictx4Cat)
        # print('-- avg\n', predictx4_avg)

        out = self.upsample4x(out)
        predict_final = self.predict(out)

        return predictx4s, predictx4_avg, predict_final

class InvLoss(nn.Module):
    def __init__(self, lamda=0.5):
        super(InvLoss, self).__init__()
        self.lamda = lamda

    def forward(self, _input, _target):
        dArr = _input - _target

        mseLoss = torch.sum(torch.sum(dArr*dArr, 2), 3)/nVal
        dArrSum = torch.sum(torch.sum(dArr, 2), 3)
        mssLoss = -self.lamda*(dArrSum*dArrSum)/(nVal**2)

        loss = mseLoss + mssLoss
        loss = torch.sum(loss)
        return loss

        