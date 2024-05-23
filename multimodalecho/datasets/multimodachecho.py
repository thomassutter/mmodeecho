"""Dataloader"""

import os

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import multimodalecho


class Multimodalecho(Dataset):
    def __init__(self, data_dir, datatype, frames, num_modes, in_channels, axis, width, feature_selection, \
         transform=None, split="all"):
        self.data_dir = data_dir
        self.datatype = datatype
        self.frames = frames
        self.num_modes = num_modes
        self.width = width
        self.transform = transform
        
        with open(os.path.join(data_dir, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        
        # Remove videos which do not have the required NumberOfFrames to generate M-mode of desired width
        data = data.loc[data["NumberOfFrames"] >= width, :]

        if split != "all":
            data = data[data["Split"] == split].reset_index()

        self.fnames = data["FileName"].to_numpy()
        self.EF = torch.tensor(data["EF"].values, dtype=torch.float)

        # Generate M-modes / frames at initialisation to avoid re-generating them each epoch
        self.input = torch.empty((len(self.fnames), in_channels, 112, width))

        # tensors containing pre-extracted M-modes
        if datatype == "tensors":
            # load first tensor to determine angles to keep
            tensor = multimodalecho.utils.load_tensor(os.path.join(self.data_dir, "Mmodes", self.fnames[0]))
            # select the closest angles from tensor containing high number of M-modes
            num_modes_tensor = tensor.size(0)
            tensor_angles = multimodalecho.utils.compute_angles(0, num_modes_tensor)
            new_angles = multimodalecho.utils.compute_angles(0, self.num_modes).astype(np.float64)
            # increase new_angles for searchsorted (side=left)
            new_angles -= 90 / num_modes_tensor
            angle_indices = np.searchsorted(tensor_angles, new_angles)
            self.input[0, :, :, :] = tensor[angle_indices]

            # select angles for remaining samples
            for index, filename in enumerate(self.fnames[1:]):
                tensor = multimodalecho.utils.load_tensor(os.path.join(self.data_dir, "Mmodes", filename))
                # select the closest angles from tensor containing high number of M-modes
                self.input[index+1, :, :, :] = tensor[angle_indices]

        elif datatype == "videos":
            data["FileName"] = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix

            # select individual frames from each video
            if self.frames:
                for index, filename in enumerate(fnames):
                    video = multimodalecho.utils.load_video(os.path.join(self.data_dir, "Videos", filename))
                    # sample requested number of random frames and extract them
                    to_extract = np.sort(np.random.randint(1, self.width, self.num_modes))
                    frames = video[to_extract, :, :]
                    self.input[index, :, :, :] = torch.from_numpy(frames)

            else: # M-mode
                # Need volume tracings (long axis, i.e. first entry) for informed axis
                if axis == "informed":
                    with open(os.path.join(data_dir, "VolumeTracings.csv")) as f:
                        tracings = pandas.read_csv(f)
                        # need to keep only the first entry per (FileName, Frame) pair, the rest are short axis tracings
                        tracings = tracings.drop_duplicates(subset=["FileName", "Frame"])
                        tracings = tracings.drop("Frame", axis=1)

                        # tracings are unavailable for some samples -> remove these samples
                        data = pandas.merge(data, tracings, how="inner", on="FileName").drop_duplicates(subset=["FileName"])
                        data = data.drop(["X1", "Y1", "X2", "Y2"], axis=1)
                        
                self.fnames = data["FileName"].to_numpy()
                self.EF = torch.tensor(data["EF"].values, dtype=torch.float)
                self.input = torch.empty((len(self.fnames), in_channels, 112, width))
                
                with tqdm(self.fnames, desc=f"Generating M-modes {split}", unit="files") as fnames:
                    for index, filename in enumerate(fnames):
                        if axis == "informed":
                            trcs = tracings.loc[tracings["FileName"] == filename, :]
                            trcs = trcs.drop("FileName", axis=1).values
                            # take the mean of the two tracings to approximate long axis of the heart
                            trcs = np.mean(trcs, axis=0)
                            vector = trcs[:2] - trcs[2:]
                            # find the center of the LV
                            center = np.round(trcs[:2] - vector/2).astype(np.int8)
                            # normalise the vector to calculate angle offset
                            vector = vector / np.linalg.norm(vector)

                            # calculate angle between vector and vertical line
                            offset = np.degrees(np.arccos(np.dot(vector, np.array([0, 1])))) 
                            offset = offset - 180

                        elif axis == "random_start":
                            center = None
                            offset = np.random.randint(0, 360, 1)[0]

                        else: # default or random
                            center = None
                            offset = 0

                        path = os.path.join(data_dir, "Videos", filename)
                        video = multimodalecho.utils.load_video(path) # format is [#frames, height, width]
                        video = video[:width, :, :] # cut frames at the end
                        self.input[index, :, :, :] = multimodalecho.utils.generate_Mmode(video, \
                            axis, num_modes, center, offset, feature_selection)

        elif self.datatype == "images":
            for index, filename in enumerate(fnames):
                self.input[index, :, :, :] = multimodalecho.utils.load_images(os.path.join(self.data_dir, filename))

        else:
            print("The datatype you requested cannot be handled.\n"
                   "Please select one of {'tensors', 'images', 'videos'}.")
            raise NotImplementedError

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index: int):
        input = self.input[index, :, :, :]
        input = input.to(torch.float) # change dtype to allow gradient computations
        label = self.EF[index]

        return input, label



