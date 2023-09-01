"""Dataset for M-mode Echocardiography"""

import os
import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

class MModeEcho(Dataset):
    def __init__(
            self, 
            data_dir="None", 
            num_modes=10, 
            sample_mode="random", 
            nb_thresh=50, 
            aug="mix", 
            percent=-1.0, 
            split="TRAIN", 
            num_clips=1):
        """ M-mode Echocardiography Dataset
        Args:
            data_dir: path to the directory containing the M-modes
            num_modes: number of modes to use for each patient
            sample_mode: how to select the M-mode images of each patient ({"sequential", "fixed_random", "random"})
            nb_threshold: threshold for the distance of the M-mode images to be considered as neighbors
            aug: data augmentation method ({"mix", "none"})
            percent: percentage of data to use (default: -1.0, use all patients)
            split: split of the dataset to use ({"TRAIN", "VAL", "TEST", "all"})
            num_clips: number of random clips to sample from each patient (set to None to use one fixed long clip)
        """

        self.data_dir = data_dir
        self.num_modes = num_modes
        self.split = split
        self.sample_mode = sample_mode
        self.nb_thresh = nb_thresh

        if num_clips is None:
            self.suffix = "Mmodes_50"
            self.num_clips = 1
            self.max_clips = 1
            self.width = 112
        else:
            self.suffix = "Mmodes_c10"
            self.num_clips = num_clips
            self.max_clips = 10
            self.width = 32
        assert self.num_clips <= self.max_clips
        
        if aug is None:
            self.transform = None
        elif aug == "mix":
            self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.8),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.8)
            ])
        else:
            raise NotImplementedError
        with open(os.path.join(data_dir, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        
        # Remove videos which do not have the required NumberOfFrames to generate M-mode of desired width
        data = data.loc[data["NumberOfFrames"] >= self.width, :]

        if split != "all":
            data = data[data["Split"] == split].reset_index()
        
        self.fnames = data["FileName"].to_numpy()
        self.EF = data["EF"].values

        if percent > 0 and percent <= 1: # use a percentage of the data
            nb_samples = int(len(self.fnames) * percent)
            shuffler = np.random.permutation(len(self.fnames))
            self.fnames = self.fnames[shuffler][:nb_samples]
            self.EF = self.EF[shuffler][:nb_samples]

        self.EF = torch.tensor(self.EF, dtype=torch.float)
        self.input = torch.empty((len(self.fnames), self.num_clips, num_modes, 1, 112, self.width))
        if self.transform is not None:
            self.aug_input = torch.empty((len(self.fnames), self.num_clips, num_modes, 1, 112, self.width))

        if sample_mode == "sequential":
            assert num_modes <= 50
            for index, filename in enumerate(self.fnames):
                tensor = self.tensor_unify(load_tensor(os.path.join(data_dir, self.suffix, filename)))
                if self.transform is not None:
                    aug_tensor = self.transform(tensor)
                for i in range(num_modes):
                    self.input[index, :, i, 0, :, :] = tensor[:num_clips, i:i+1, :, :]
                    if self.transform is not None:
                        self.aug_input[index, :, i, 0, :, :] = aug_tensor[:num_clips, i:i+1, :, :]
        elif sample_mode == "fixed_random":
            # fixed for each epoch
            for index, filename in enumerate(self.fnames):
                tensor = self.tensor_unify(load_tensor(os.path.join(data_dir, self.suffix, filename)))
                if self.transform is not None:
                    aug_tensor = self.transform(tensor)
                anchor = np.random.choice(50, 1)
                left = max(0, anchor - self.nb_thresh)
                right = min(50, anchor + self.nb_thresh)
                if self.num_clips == self.max_clips:
                    clip_index = np.arange(0, self.max_clips)
                else:
                    clip_index = np.random.choice(np.arange(0, self.max_clips), self.num_clips, replace=False)
                modes_index = np.random.choice(np.arange(left, right), num_modes, replace=False)
                self.input[index, :, :, 0, :, :] = tensor[clip_index, modes_index, :, :]
                if self.transform is not None:
                    self.aug_input[index, :, :, 0, :, :] = aug_tensor[clip_index, modes_index, :, :]
        else: # "random"
            # preload all the data to save time
            self.tensor = torch.empty((len(self.fnames), self.max_clips, 50, 112, self.width))
            for index, filename in enumerate(self.fnames):
                tensor = self.tensor_unify(load_tensor(os.path.join(data_dir, self.suffix, filename)))
                self.tensor[index] = tensor

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index: int):
        if self.sample_mode == "random":     
            anchor = np.random.choice(50, 1)
            left = max(0, anchor - self.nb_thresh)
            right = min(50, anchor + self.nb_thresh)
            modes_index = np.random.choice(np.arange(left, right), self.num_modes, replace=False)
            if self.num_clips == self.max_clips:
                clip_index = np.arange(0, self.max_clips)
            else:
                clip_index = np.random.choice(np.arange(0, self.max_clips), self.num_clips, replace=False)
            self.input[index, :, :, 0, :, :] = self.tensor[index, clip_index, modes_index, :, :]
            if self.transform is not None:
                self.aug_input[index, :, :, 0, :, :] = self.transform(self.tensor[index, clip_index, modes_index, :, :])

        input = self.input[index] # (num_clips, num_modes, 1, 112, width)
        input = input.to(torch.float)

        aug_input = None
        if self.transform is not None:
            aug_input = self.aug_input[index]
            aug_input = aug_input.to(torch.float)
    
        label = self.EF[index]

        return (input, aug_input, label) if aug_input is not None else (input, label)
    
    def tensor_unify(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        assert len(tensor.shape) == 4
        return tensor