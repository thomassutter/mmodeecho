"""Utility function to extract M(otion)-mode images from B(rightness)-mode videos""" 
 
import os 
import yaml
import pandas 
 
import click
import numpy as np 
import torch 
from pathlib import Path 
from tqdm import tqdm
 
import multimodalecho 
 
 
@click.command("extract_Mmode") 
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None) 
@click.option("--output", type=click.Path(file_okay=False), default=None) 
@click.option("--split", type=str, default='all') 
@click.option("--axis", type=str, default="default") 
@click.option("--num_modes", type=int, default=1) 
@click.option("--width", type=int, default=112) 
@click.option("--seed", type=int, default=0) 

def run( 
    data_dir=None, 
    output=None, 
 
    split='all', 
    axis="default", 
    num_modes=1, 
    width=112, 
    seed=0 
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

    # Load YAML config file and set any missing parameters
    try: 
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            # Set parameters not specified via command line
            if data_dir is None: data_dir = config["extract_Mmode"]["data_dir"]
            if output is None: output = config["extract_Mmode"]["output"]
            if split is None: split = config["extract_Mmode"]["split"]
            if axis is None: axis = config["extract_Mmode"]["axis"]
            if num_modes is None: num_modes = config["extract_Mmode"]["num_modes"]
            if width is None: width = config["extract_Mmode"]["width"]
            if seed is None: seed = config["extract_Mmode"]["seed"]

    except FileNotFoundError as _:
        pass
 
    # Seed RNGs 
    np.random.seed(seed) 
    torch.manual_seed(seed)
 
    # Set default output directory 
    if output is None: 
        output = os.path.join("..", "..", "output", f"extract_Mmode_{width}_{axis}_{num_modes}") 
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "Mmodes"), exist_ok=True) 
 
    # Load video-level labels 
    with open(os.path.join(data_dir, "FileList.csv")) as f: 
        data = pandas.read_csv(f) 
        # Remove videos which do not have the required NumberOfFrames to generate M-mode of desired width 
        data = data.loc[data["NumberOfFrames"] >= width, :] 
        # copy this file into new directory
        new_data = data.copy() 
        new_data = new_data[["FileName", "EF", "NumberOfFrames", "Split"]] 
        new_data.to_csv(os.path.join(output, "FileList.csv"), index=False) 
 
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
        out_path = os.path.join(output, "Mmodes", f"{Path(filename).stem}.pt")
        # check whether Mmodes have already been generated and saved
        if os.path.exists(out_path):
            continue

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
        video = multimodalecho.utils.load_video(path) # format is [#frames, height, width] 
        video = video[:width, :, :] # cut frames at the end 

        mmodes = multimodalecho.utils.generate_Mmode(video, axis, num_modes, center, \
            offset, feature_selection=None) # format is [num_modes, height, width]
        torch.save(mmodes, out_path)

        
 
