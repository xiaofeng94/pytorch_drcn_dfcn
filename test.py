import argparse

import torch
from torch.autograd import Variable
from my_models import RDCN_VGG

import scipy.io as sio
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="pythorch recusive densely-connected nerual network Test")
parser.add_argument("--model", default=None, type=str, help="model path")
parser.add_argument("--image", default=None, type=str, help="image name")
parser.add_argument("--cpu", action="store_true", help="Use cpu only")

opt = parser.parse_args()
print(opt)

test_file = opt.image
data = sio.loadmat(test_file)
data = data['data']

rgb = data['rgb'][0,0]
depth = data['depth'][0,0]
depthx4 = data['depthx4'][0,0]

rgb_new = rgb.transpose((2, 0, 1))
rgb_new = torch.from_numpy(rgb_new[np.newaxis,:,:,:]).float()
# depth_new = torch.from_numpy(depth[np.newaxis, :, :]).float()
# depthx4_new = torch.from_numpy(depthx4[np.newaxis, :, :]).float()

print('build model...')
model = torch.load(opt.model)["model"]

# print(model)

if not opt.cpu:
    model = model.cuda()
    rgb_new = rgb_new.cuda()
    # depth_new = depth_new.cuda()
    # depthx4_new = depthx4_new.cuda()

predictx4s, predictx4_avg, predict_final = model(Variable(rgb_new))
# _, predictx4_2, predict_2 = model(Variable(torch.randn(1,3,480,640).cuda()))


predictx4_avg = predictx4_avg.cpu()
predict_final = predict_final.cpu()
# predictx4_2 = predictx4_2.cpu()

depthx4 = np.exp(depthx4)
predictx4_np = np.exp( predictx4_avg.data[0].numpy()[0,...].astype(np.float32) )
predict_np = np.exp( predict_final.data[0].numpy()[0,...].astype(np.float32) )
# predict_np2 = np.exp( predictx4_2.data[0].numpy()[0,...].astype(np.float32) )

# print(predict_np)
# print(predict_np2)
# diff = predict_np-predict_np2

# print(diff)

# depthx4_avg = Image.fromarray(predict_np)
# depthx4_avg.show()

# depth_final = Image.fromarray(predict_np2)
# depth_final.show()

sio.savemat('results.mat', {'rgb': rgb, 'depthx4':depthx4, 
                            'depthx4_pred': predictx4_np, 'depth_pred': predict_np})

print('Done!')