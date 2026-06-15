import torch
import scipy.io as scio
import os
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, directory, input_key, target_key,sampled_key,mask_key):
        """
        Args:
            directory   (str): 包含.mat文件的文件夹路径。
            input_key   (str): mat文件中对应输入的键。
            sampled_key (str): mat文件中对应采样信号的键。
            target_key  (str): mat文件中对应目标(标签)的键。
            mask_key    (str): mat文件中对应掩码的键。
        """
        self.directory = directory
        self.input_key = input_key
        self.target_key = target_key
        self.sampled_key = sampled_key
        self.mask_key = mask_key
        self.files = [f for f in os.listdir(directory) if f.endswith('.mat')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        mat_data = scio.loadmat(file_path)
        
        input_signal = mat_data[self.input_key]
        sampled_signal = mat_data[self.sampled_key]
        mask = mat_data[self.mask_key]
        target_signal = mat_data[self.target_key]
        
        input_signal = torch.tensor(input_signal, dtype=torch.float32)  
        sampled_signal = torch.tensor(sampled_signal,dtype = torch.float32)
        target_signal = torch.tensor(target_signal, dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.float32)

        return input_signal, target_signal, sampled_signal, mask