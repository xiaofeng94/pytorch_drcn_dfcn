import torch.utils.data as data
import torch

import glob
import scipy.io as sio
import numpy as np

# TODO: data argumentation
class NYUv2DataSet(data.Dataset):
    def __init__(self, data_root, is_train=True):
        super(NYUv2DataSet, self).__init__()

        self.dataRoot = data_root
        self.dataFiles = glob.glob('%s/*.mat'%self.dataRoot)
        self.dataNum = len(self.dataFiles)
        self.requiredSize = [240, 320]
        self.reqSizex4 = [60, 80]
        self.isTrain = is_train

    def __getitem__(self, index):
        currFile = self.dataFiles[index]
        data = sio.loadmat(currFile)
        data = data['data']

        rgb = data['rgb'][0,0].transpose((2, 0, 1))
        depth = data['depth'][0,0]
        depthx4 = data['depthx4'][0,0]
        imageSize = data['imageSize'][0,0][0]
        # print(imageSize)
        if imageSize[0] < self.requiredSize[0] or imageSize[1] < self.requiredSize[1]:
            raise ValueError('input image size is smaller than [240, 320]')

        if self.isTrain:
            import random
            offset_x = random.randint(0, imageSize[0] - self.requiredSize[0]) // 4
            offset_y = random.randint(0, imageSize[1] - self.requiredSize[1]) // 4
        else:
            offset_x = int((imageSize[0] - self.requiredSize[0])/2) // 4
            offset_y = int((imageSize[1] - self.requiredSize[1])/2) // 4

        rgb = rgb[:, 4*offset_x:4*offset_x+self.requiredSize[0],
                        4*offset_y:4*offset_y+self.requiredSize[1]]
                        
        depth = depth[np.newaxis, 4*offset_x:4*offset_x+self.requiredSize[0],
                                        4*offset_y:4*offset_y+self.requiredSize[1]]

        depthx4 = depthx4[np.newaxis, offset_x:offset_x+self.reqSizex4[0],
                                        offset_y:offset_y+self.reqSizex4[1]]

        return torch.from_numpy(rgb).float(), torch.from_numpy(depth).float(), torch.from_numpy(depthx4).float()

        # A_path = self.A_paths[index]

        # data = sio.loadmat(A_path)
        # data = data['data']
        # rgb = data['rgb'][0,0]
        # depth = data['depth'][0,0]

        # # crop image to fineSize(256 for default)
        # offset = max(0, math.floor((self.osize - self.fineSize)/2)) # random.randint(0, self.osize-self.fineSize)
        # rgb_crop = rgb[offset:offset+self.fineSize, offset:offset+self.fineSize, :]
        # depth_crop = depth[offset:offset+self.fineSize, offset:offset+self.fineSize]

        # # fill depth values in all channels
        # depth_final = np.ones((self.fineSize, self.fineSize, 3))
        # depth_final[:,:,0] = depth_crop
        # depth_final[:,:,1] = depth_crop
        # depth_final[:,:,2] = depth_crop

        # rgb_fianl = rgb_crop.transpose((2, 0, 1))
        # depth_final = depth_final.transpose((2, 0, 1))

        # return {'A': rgb_fianl, 'B': depth_final,
        #         'A_paths': A_path, 'B_paths': A_path}

    def __len__(self):
        return self.dataNum

