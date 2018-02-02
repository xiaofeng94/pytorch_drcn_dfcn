### used to convert .caffemodel to .pth
import sys
caffe_root = '/media/xiaofeng/codes/LinuxFiles/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe

import numpy as np

caffe_cfg = '/media/xiaofeng/learning/DeepLearning/Caffe/caffe/models/cnn-models/VGG19_cvgj/deploy.prototxt'
caffemodel_path = '/media/xiaofeng/learning/DeepLearning/Caffe/caffe/models/vgg19_cvgj_iter_300000.caffemodel'

target_path = './model/pretrain/vgg19_cvgj'


caffe.set_mode_cpu()
net = caffe.Net(caffe_cfg, caffemodel_path, caffe.TEST)

model_dict = dict()
for key in net.params.keys():
    if 'conv' in key:
        print('-- {}, len: {}'.format(key, len(net.params[key])))
        for indx in range(len(net.params[key])):
            currKey = '%s_%d'%(key, indx)
            shape = net.params[key][indx].data.shape
            data = net.params[key][indx].data

            model_dict[currKey] = data

    # if key == 'conv1_1':
    #     print(net.params[key][0].data)
    #     print(net.params[key][1].data)

import scipy.io as sio

# sio.savemat('conv1_1', {'conv1_1': model_dict['conv1_1'][0]})
print('save model to %s'%target_path)
sio.savemat(target_path, model_dict)