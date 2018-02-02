import torch.utils.data as data
import torch

import glob
import scipy.io as sio
import numpy as np

# TODO: data argumentation
class NYUv2DataSet(data.Dataset):
    def __init__(self, data_root):
        super(NYUv2DataSet, self).__init__()

        self.dataRoot = data_root
        self.dataFiles = glob.glob('%s/*.mat'%self.dataRoot)
        self.dataNum = len(self.dataFiles)

    def __getitem__(self, index):
        currFile = self.dataFiles[index]
        data = sio.loadmat(currFile)
        data = data['data']

        rgb = data['rgb'][0,0].transpose((2, 0, 1))
        depth = data['depth'][0,0]
        depthx4 = data['depthx4'][0,0]

        depth = depth[np.newaxis, :, :]
        depthx4 = depthx4[np.newaxis, :, :]

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

