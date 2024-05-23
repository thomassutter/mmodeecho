"""Utility function to extract videos of sufficient length and cut to said length""" 
 
import os
import yaml
import pandas 
 
import click
import numpy as np 
import torch 
from tqdm import tqdm
 
import multimodalecho 
 
 
@click.command("extract_videos") 
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None) 
@click.option("--output", type=click.Path(file_okay=False), default=None) 
@click.option("--split", type=str, default=None) 
@click.option("--length", type=int, default=None) 

def run( 
    data_dir=None, 
    output=None, 
 
    split=None, 
    length=None, 
): 
    """ 
    Args: 
        data_dir (string): data_dir directory of dataset (defaults to `echonet.config.DATA_DIR`) 
        output (string): output directory 
        split (string): One of {"all", "TRAIN", "VAL", "TEST"} 
            (defaults to "all") 
        length (int): determines the minimum length videos have to have 
            and length to which videos are cut to
            (defaults to 112) 
         
    Returns: 
        None. 
    """ 

    # Load YAML config file and set any missing parameters
    try: 
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            # Set parameters not specified via command line
            if data_dir is None: data_dir = config["extract_videos"]["data_dir"]
            if output is None: output = config["extract_videos"]["output"]
            if split is None: split = config["extract_videos"]["split"]
            if length is None: length = config["extract_videos"]["length"]

    except FileNotFoundError as _:
        pass
 
    # Set default output directory 
    if output is None: 
        output = os.path.join("..", "..", f"extract_videos_{length}") 
    os.makedirs(output, exist_ok=True) 
    os.makedirs(os.path.join(output, "Videos"), exist_ok=True)
 
    # Load video-level labels 
    with open(os.path.join(data_dir, "FileList.csv")) as f: 
        data = pandas.read_csv(f) 
        # Remove videos which do not have the required NumberOfFrames (length)
        data = data.loc[data["NumberOfFrames"] >= length, :] 
        
        # copy this file into new directory, adjusting NumberOfFrames and FPS 
        new_data = data.copy() 
        new_data = new_data[["FileName", "EF", "NumberOfFrames", "Split", "FPS"]] 
        new_data.to_csv(os.path.join(output, "FileList.csv"), index=False) 
 
    data["Split"].map(lambda x: x.upper()) 
 
    if split != "all": 
        data = data[data["Split"] == split] 
 
    fps = data["FPS"].tolist()
    fnames = data["FileName"].tolist() 
    fnames = [fn + ".avi" for fn in fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix 
 
    # Copy volume tracings to new directory
    with open(os.path.join(data_dir, "VolumeTracings.csv")) as f:
        tracings = pandas.read_csv(f)
        # drop fnames with insufficient width
        tracings = tracings[tracings["FileName"].isin(fnames)]
        tracings.to_csv(os.path.join(output, "VolumeTracings.csv"), index=False)

    # Check that all video files are present 
    missing = set(fnames) - set(os.listdir(os.path.join(data_dir, "Videos"))) 
    if len(missing) != 0: 
        print("{} videos could not be found in {}:".format( 
            len(missing), os.path.join(data_dir, "Videos"))) 
        for f in sorted(missing): 
            print("\t", f) 
        raise FileNotFoundError(os.path.join(data_dir, "Videos", sorted(missing)[0]))  
 
    # Load videos, cut them to equal length, save them 
    for filename, f in tqdm(zip(fnames, fps), total=len(fnames), desc="Extracting videos of desired length", unit="videos"): 
        # check whether video has already been cut and saved
        output_path = os.path.join(output, "Videos", filename)
        if os.path.exists(output_path):
            continue
 
        path = os.path.join(data_dir, "Videos", filename) 
        video = multimodalecho.utils.load_video(path).astype(np.uint8) # format is [#frames, height, width] 
        video = video[:length, :, :] # cut frames at the end 
        multimodalecho.utils.savevideo(output_path, video, fps=f) 
