import argparse, os
import math, random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from my_models import RDCN_VGG
from datasets import NYUv2DataSet

parser = argparse.ArgumentParser(description="pytorch recusive densely-connected nerual network")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cpu", action="store_true", help="Use cpu only")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--disp", type=int, default=5, help="display interval, Default:5")
parser.add_argument("--alpha", type=int, default=0.5, help="the coefficient for x4 loss, Default: 0.5")
parser.add_argument("--lamda", type=int, default=1, help="the coefficient for complete loss, Default: 1")
parser.add_argument("--rec_num", type=int, default=1, help="recusive times that the data go through the dense block")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--debug", action="store_true", help="debug network parameters")


def main():
    print('start to train..')
    global opt, model

    opt = parser.parse_args()
    print(opt)

    cpu_only = opt.cpu  

    if (not cpu_only) and not torch.cuda.is_available():
        raise Exception("No GPU found, , please run with --cpu")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if not cpu_only:   
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print('build model...')
    model = RDCN_VGG(opt.rec_num)
    criterion = nn.MSELoss(size_average=True)
    print_network(model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    if not cpu_only:
        print("Setting GPU")
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)

    train_set = NYUv2DataSet('/media/xiaofeng/learning/NYUv2_DATA/Dataset/rdcn_crop/train')
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    cumulated_loss = 0
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print ("epoch = %d, lr = %.12f"%(epoch,optimizer.param_groups[0]["lr"]))
    model.train()

    for iterCount, batch in enumerate(training_data_loader, 1):
        inputData, target, targetx4 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        if not opt.cpu:
            inputData = inputData.cuda()
            target = target.cuda()
            targetx4 = targetx4.cuda()

        predictx4s, predictx4_avg, predict_final = model(inputData)

        x4Loss = None
        for indx in range(len(predictx4s)):
            if x4Loss == None:
                x4Loss = criterion(predictx4s[indx], targetx4)
            else:
                x4Loss = x4Loss + criterion(predictx4s[indx], targetx4)
        x4Loss = x4Loss/len(predictx4s)
        avgLoss = criterion(predictx4_avg, targetx4)
        finalLoss = criterion(predict_final, target)

        loss = opt.alpha*x4Loss + (1-opt.alpha)*avgLoss + opt.lamda*finalLoss
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # debug grad
        if opt.debug:
            if loss.data[0] < 0.4:
                count = 0
                for param in model.parameters():
                    count += 1
                    gradSum = torch.sum(param.grad)
                    # if abs(gradSum.data[0]) < 1e-6:
                    print('-- {}, shape: {} ---------------'.format(count, param.shape))
                    print(gradSum.data[0])

                input('end of the step')
        
        if iterCount%opt.disp == 0:
            print("Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iterCount, len(training_data_loader), cumulated_loss/opt.disp))
            cumulated_loss = 0
        else:
            cumulated_loss += loss.data[0]

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()