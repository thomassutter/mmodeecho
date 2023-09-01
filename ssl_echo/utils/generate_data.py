import os 
import pandas 
import numpy as np 
import torch 
from pathlib import Path 
from tqdm import tqdm 
from utils import generate_Mmode, load_video


def run( 
    data_dir="/cluster/work/vogtlab/Projects/EchoNet/", 
    output="/cluster/work/vogtlab/Projects/EchoNet-Mmode/", 
 
    split='all', 
    axis="default", 
    num_modes=50, 
    width=32, 
    seed=0,
    clips=10,
    period=2,
): 
    """Loads all available videos from all splits, 
       generates the desired number of M-mode images, 
       saves them in one folder per original video. 
 
    Args: 
        data_dir (string): data_dir directory of dataset (defaults to `echonet.config.DATA_DIR`) 
        output (string): output directory 
        split (string): One of {"all", "TRAIN", "VAL", "TEST"} 
            (defaults to "all") 
        axis (string): One of {"default", "informed", "random"} 
            (defaults to "default") 
        num_modes (int): determines the number of M-mode images generated 
            (defaults to 1) 
        width (int): determines the width of the resulting M-mode images 
            (defaults to 112) 
        seed (int, optional): Seed for random number generator (defaults to 0) 
         
    Returns: 
        None. 
    """ 
    # Seed RNGs 
    ct = 0
    np.random.seed(seed) 
    torch.manual_seed(seed)
 
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "Mmodes_clip"), exist_ok=True) 
 
    # Load video-level labels 
    with open(os.path.join(data_dir, "FileList.csv")) as f: 
        data = pandas.read_csv(f) 
        # Remove videos which do not have the required NumberOfFrames to generate M-mode of desired width 
        data = data.loc[data["NumberOfFrames"] >= (clips + (width - 1) * period), :] 
        # copy this file into new directory
        # new_data = data.copy() 
        # new_data = new_data[["FileName", "EF", "NumberOfFrames", "Split"]] 
        # new_data.to_csv(os.path.join(output, "FileList.csv"), index=False) 
 
    data["Split"].map(lambda x: x.upper()) 
 
    if split != "all": 
        data = data[data["Split"] == split] 
 
    fnames = data["FileName"].tolist()
    fnames = [fn + ".avi" for fn in fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix 
 
    # Check that all video files are present 
    missing = set(fnames) - set(os.listdir(os.path.join(data_dir, "Videos"))) 
    if len(missing) != 0: 
        print("{} videos could not be found in {}:".format( 
            len(missing), os.path.join(data_dir, "Videos"))) 
        for f in sorted(missing): 
            print("\t", f) 
        raise FileNotFoundError(os.path.join(data_dir, "Videos", sorted(missing)[0]))  
 
    # Need volume tracings (long axis, i.e. first entry) for informed axis 
    if axis == "informed": 
        with open(os.path.join(data_dir, "VolumeTracings.csv")) as f: 
            tracings = pandas.read_csv(f) 
            # need to keep only the first entry per (FileName, Frame) pair, the rest are short axis tracings 
            tracings = tracings.drop_duplicates(subset=["FileName", "Frame"]) 
            tracings = tracings.drop("Frame", axis=1) 
 
    # Load videos, cut them to equal length, generate M-modes 
    for filename in tqdm(fnames, desc=f"Extracting {num_modes} M-modes per video", unit="videos"): 
        out_path = os.path.join(output, "Mmodes_c10", f"{Path(filename).stem}.pt")
        # check whether Mmodes have already been generated and saved 
        # if os.path.exists(out_path):
        #     continue

        if axis == "informed": 
            trcs = tracings.loc[tracings["FileName"] == filename, :].drop("FileName", axis=1).values 
            # take the mean of the two tracings to approximate long axis of the heart 
            trcs = np.mean(trcs, axis=0) 
            vector = trcs[:2] - trcs[2:] 
            # find the center of the LV
            center = np.round(trcs[:2] - vector/2).astype(np.int8)
            # normalise the vector to calculate angle offset
            vector = vector / np.linalg.norm(vector) 
 
            offset = np.degrees(np.arccos(np.dot(vector, np.array([0, 1])))) # calculate angle between vector and vertical line 
            offset = offset - 180 

        elif axis == "random_start":
            center = None
            offset = np.random.randint(0, 360, 1)
 
        else: # default or random 
            center = None
            offset = 0 
 
        path = os.path.join(data_dir, "Videos", filename) 
        video = load_video(path) # format is [#frames, height, width]
        f, h, w = video.shape
        start = np.random.choice(f - (width - 1) * period, clips)
        videos = tuple(video[s + period * np.arange(width), :, :] for s in start)
        # video = video[:width, :, :] # cut frames at the end 

        mmodes_clips = []
        for i in range(clips):
            mmodes = generate_Mmode(videos[i], axis, num_modes, center, \
                offset, feature_selection=None) # format is [num_modes, height, width]
            mmodes_clips.append(mmodes)
        mmodes_clips = torch.stack(mmodes_clips)
        if ct == 0:
            print(mmodes_clips.shape)
            ct = 1
        torch.save(mmodes_clips, out_path)

if __name__ == "__main__":
    run()