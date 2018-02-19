import argparse, os
import math, random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from my_models import DFCN_32, InvLoss
from datasets import NYUv2DataSet

parser = argparse.ArgumentParser(description="pytorch recusive densely-connected nerual network")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--step_size", type=int, default=8, help="update the parameters when n steps go")
parser.add_argument("--nEpochs", type=int, default=800, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_step", type=int, default=80, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=80")
parser.add_argument("--cpu", action="store_true", help="Use cpu only")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--disp", type=int, default=16, help="display interval, Default:16")
# parser.add_argument("--alpha", type=int, default=0.5, help="the coefficient for x4 loss, Default: 0.5")
# parser.add_argument("--lamda", type=int, default=1, help="the coefficient for complete loss, Default: 1")
# parser.add_argument("--rec_num", type=int, default=1, help="recusive times that the data go through the dense block")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--debug", action="store_true", help="debug network parameters")
parser.add_argument("--pretrain", default="", type=str, help="use pretrained model to initilize convolution")
parser.add_argument("--save_dir", default="model/", type=str, help="asign the save dir of checkpoint")
parser.add_argument("--save_step", type=int, default=10,  help="save model every assigned epochs")
parser.add_argument("--data", default="/media/xiaofeng/learning/NYUv2_DATA/Dataset/rdcn_crop/train", type=str, help="training data folder")
parser.add_argument("--net_type", default="dfcn_32", type=str, help="assign what kind of network to be used, default: dfcn_32")
parser.add_argument("--lamda", type=int, default=0.5, help="the coefficient for sum square part in invLoss, Default: 0.5")

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
    if opt.net_type == 'dfcn_32':
        model = DFCN_32()
    else:
        print('model type %s does not exist'%(opt.net_type))
        exit(-1)
    print_network(model)

    # use pretrained vgg model
    if opt.pretrain:
        print('load pretrained model...')
        model.loadConv(opt.pretrain)
        # input('done !!')

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
        # criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)

    train_set = NYUv2DataSet(opt.data)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, epoch)
        if epoch < 10 or epoch%opt.save_step == 0:
            save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.4 ** (epoch // opt.lr_step))
    return lr    

# def invLoss(input, target, lamda=0.5):
#     dArr = input - target

#     mseLoss = torch.sum(torch.sum(dArr*dArr, 2), 3)/nVal
#     dArrSum = torch.sum(torch.sum(dArr, 2), 3)
#     mssLoss = -lamda*(dArrSum*dArrSum)/(nVal**2)

#     loss = mseLoss + mssLoss
#     loss = torch.sum(loss)
#     return loss


def train(training_data_loader, optimizer, model, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print ("epoch = %d, lr = %.12f"%(epoch,optimizer.param_groups[0]["lr"]))
    model.train()

    cumulated_loss = 0
    stepCount = 0
    optimizer.zero_grad()
    # criterion = nn.MSELoss(size_average=True)
    criterion = InvLoss(opt.lamda)

    for iterCount, batch in enumerate(training_data_loader, 1):
        inputData, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if not opt.cpu:
            inputData = inputData.cuda()
            target = target.cuda()
            targetx4 = targetx4.cuda()

        predicted_depth = model(inputData)

        loss = criterion(predicted_depth, target)

        if stepCount >= opt.step_size:
            stepCount = 0
            optimizer.zero_grad()

        loss.backward()
        stepCount += 1
        iterCount += 1

        if stepCount >= opt.step_size or iterCount >= len(training_data_loader):
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
    model_out_path = opt.save_dir + "/model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()