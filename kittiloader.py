import torch.utils.data as data

import os
import glob
import numpy as np

import torch
import torchvision

def read_pc(filename):
    '''
    Read point cloud to ndarray from a TXT file
    The TXT file contains pre-processed point cloud using PCL (https://pointclouds.org/)
    One line of the TXT file represents: [x y z intensity nx ny nz curvature]
    '''
    scan = np.loadtxt(filename, dtype=np.float32)
    # drop intensity
    pc = scan[:,0:3]
    sn = scan[:,4:8]
    return pc, sn

def read_calib(filename):
    '''
    Read camera to velodyne tranformation matrix from calibration file
    Output:
        Tr: 4X4 Transformation matrix
    '''
    with open(filename) as f:
        lines = f.readlines()   # 将文件中的所有行的内容存到list中，可指定保存的行数
        Tr_line = lines[-1]  # last line of datas is Tr Matrix
        Tr_words = Tr_line.split(' ')  # 12 words
        calib_list = Tr_words[1:]  # drop "Tr:"
        for i in range(len(calib_list)):
            calib_list[i] = float(calib_list[i])    # string to float
        calib_array = np.array(calib_list).astype(np.float32)
        Tr = np.zeros((4,4),dtype=np.float32)
        Tr[0,:] = calib_array[0:4]
        Tr[1,:] = calib_array[4:8]
        Tr[2,:] = calib_array[8:12]
        Tr[3,3] = 1.0
    return Tr

def read_pose(filename, Tr):
    '''
    Read vehicle pose from pose file and calibrate to velodyne frame
    
    Input:
        Tr: Transformation matrix
    '''
    Tr_inv = np.linalg.inv(Tr)  #  numpy linear algrithm: Inverse
    Tlist = []
    poses = np.loadtxt(filename, dtype=np.float32)  # 整个sequence的pose都加载进来了。poses dimension: Nx12
    for i in range(poses.shape[0]):
        one_pose = poses[i,:]
        Tcam = np.zeros((4,4),dtype=np.float32)
        Tcam[0,:] = one_pose[0:4]
        Tcam[1,:] = one_pose[4:8]
        Tcam[2,:] = one_pose[8:12]
        Tcam[3,3] = 1.0
        Tvelo = Tr_inv.dot(Tcam).dot(Tr)
        Tlist.append(Tvelo)
    return Tlist

# 将原始点云中的6万多个点随机采样成16384个点
def points_sample(points, sn, npoints):
    '''
    Random selected npoints points
    '''
    size = points.shape[0]
    new_points = np.zeros((npoints, 3))
    new_sn = np.zeros((npoints, 4))
    for i in range(npoints):
        index = np.random.randint(size)
        new_points[i,:] = points[index,:]
        new_sn[i,:] = sn[index,:]
    return new_points, new_sn

def get_pointcloud(filename, npoints):
    '''
    Read point cloud from file and random sample points
    '''
    pc, sn = read_pc(filename)
    pc, sn = points_sample(pc, sn, npoints)
    pc = torch.from_numpy(pc.astype(np.float32))
    sn = torch.from_numpy(sn.astype(np.float32))
    return pc, sn

# inherit torch datatype: data.Dataset
class KittiDataset(data.Dataset):
    def __init__(self, root, seq, npoints):
        super(KittiDataset, self).__init__()
        self.velodyne_path = os.path.join(root, 'sequences', seq, 'velodyne_txt')   # 点云文件夹的路径
        #  返回当前路径下的所有txt，保存在list中, type: list
        self.velodyne_names = glob.glob(os.path.join(self.velodyne_path, '*.txt'))  # 点云文件的路径
        self.velodyne_names = sorted(self.velodyne_names)
        # 真值位姿和外参文件的路径，Oxford和indoor数据没有外参文件
        self.poses_path = os.path.join(root, 'poses', seq+'.txt')
        self.calib_path = os.path.join(root, 'sequences', seq, 'calib.txt')
        self.npoints = npoints
        # return T Matrix
        Tr = read_calib(self.calib_path)
        # return poses in lidar frame
        self.Tlist = read_pose(self.poses_path, Tr)
        # 构造pairs，暂时还未将点云真正输入，只是构造了索引对
        self.dataset = self.make_dataset()

    # make pairs of indices
    # 所有数据集都可借鉴该法进行pair的构建
    def make_dataset(self):
        max_ind = len(self.velodyne_names)
        dataset = []
        bias = 10
        for i in range(max_ind):
            src_idx = i
            if i + bias >= max_ind:
                dst_idx = i - bias
            else:
                dst_idx = i + bias
            dataset.append([src_idx, dst_idx])  # Nx2
        return dataset
    
    # __init__执行完成后默认执行，根据上面make_dataset生成的pairs将点云数据实际加载进来
    def __getitem__(self, index):
        src_idx, dst_idx = self.dataset[index]
        src_file_name = self.velodyne_names[src_idx]
        dst_file_name = self.velodyne_names[dst_idx]
        src_pc, src_sn = get_pointcloud(src_file_name, self.npoints)
        dst_pc, dst_sn = get_pointcloud(dst_file_name, self.npoints)
        src_T = self.Tlist[src_idx] # 每帧点云的pose按照src和dst分别存入两个list
        dst_T = self.Tlist[dst_idx]
        # 获得src和dst帧点云的相对变换矩阵T，后面配准的结果与这个真值做对比，验证精度
        relaT = np.linalg.inv(dst_T).dot(src_T)
        relaT = torch.from_numpy(relaT.astype(np.float32))

        return src_pc, src_sn, dst_pc, dst_sn, relaT
    
    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    root = ''
    seq = '00'
    npoints = 16384
    trainset = KittiDataset(root, seq, npoints)
    print(len(trainset))
    src_pc, src_sn, dst_pc, dst_sn, relaT = trainset[0]
    print(src_pc.shape, src_sn.shape, dst_pc.shape, dst_sn.shape, relaT.shape)
    print(relaT)