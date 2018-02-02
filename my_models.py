from collections import OrderedDict

import torch
import torch.nn as nn

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
        return self.transition(out)

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


class RDCN_VGG(nn.Module):
    def __init__(self, rec_num):
        super(RDCN_VGG, self).__init__()

        self.recNum = rec_num
        self.downsample = nn.Sequential(OrderedDict([
            # ('data/bn', nn.BatchNorm2d(3)),
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

    def forward(self, x):
        out = self.downsample(x)
        predictx4s = [None for i in range(self.recNum)]
        predictx4Cat = None
        predict_final = None

        # input("RDCN_VGG before loop")
        for indx in range(self.recNum):
            out = self.denseBlock(out)
            predictx4s[indx] = self.predictx4(out)
            if predictx4Cat == None:
                predictx4Cat = predictx4s[indx]
            else:
                predictx4Cat = torch.cat([predictx4Cat, predictx4s[indx]], 1)

            out = self.upsample4x(out)
            out = self.predict(out)

            predict_final = out

        predictx4_avg = self.weightedAvg(predictx4Cat)

        return predictx4s, predictx4_avg, predict_final

