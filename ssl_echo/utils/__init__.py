"""Utility functions for videos, plotting and computing performance metrics."""

import os
import typing

import cv2  # pytype: disable=attribute-error
#cv2.setNumThreads(0)
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from PIL import Image

from . import loss
from . import lr_scheduler
from . import saver

def generate_Mmode(video, axis, num_modes, center, offset, feature_selection):
    video = video.astype(np.uint8)
    
    if axis == "default" or axis == "informed" or axis == "random_start":
        angles = compute_angles(offset, num_modes)

    elif axis == "random":
        angles = np.sort(np.random.randint(-90, 90, num_modes))

    else:
        print("The axis you selected is not implemented.\n \
            Please choose one of {default, informed, random_start, random}")
        raise NotImplementedError

    # remove features (images) specified
    if feature_selection is not None:
        angles = np.delete(angles, feature_selection)

    images = []

    # depending on number of modes, define evenly-spaced (rotated) axes
    for angle in angles:
        if center is not None:
            # pad the image such that center is the new center pixel
            pad_x = [video.shape[2] - center[0], center[0]]
            pad_y = [video.shape[1] - center[1], center[1]]
            padded_video = np.pad(video, [[0, 0], pad_y, pad_x], mode="constant")
            # rotate padded video
            rotated_video = ndimage.rotate(padded_video, angle, axes=(1, 2), reshape=False)
            # remove padding after rotation
            rotated_video = rotated_video[:, pad_y[0]:-pad_y[1], pad_x[0]:-pad_x[1]]

            # save_video(os.path.join(f"mode{angle}.avi"), rotated_video, fps=50)
            # save_video(os.path.join(f"mode_default{angle}.avi"), video, fps=50)

        else: # rotate around default center
            rotated_video = ndimage.rotate(video, angle, axes=(1, 2), reshape=False) 

        # extract vertical line from rotated video
        width = rotated_video.shape[2]
        # why not use vertical line passing the center?
        image = rotated_video[:, :, width // 2]
        image = image.transpose(1, 0) # want [height, width]
        # image2 = video[:, :, width // 2]
        # image2 = image2.transpose(1, 0) # want [height, width]
        # img = Image.fromarray(image, 'L')   
        # img.save(os.path.join(f"Mmode{angle}.jpg"))
        # img2 = Image.fromarray(image2, 'L')   
        # img2.save(os.path.join(f"Mmode_default{angle}.jpg"))
        image = torch.from_numpy(image)
        images.append(image)

    images = torch.stack(images)  # format is [num_modes, height, width]

    return images

def compute_angles(offset, num_modes):
    if num_modes % 2 == 0:
        start = offset - 90
        end = offset + 90
        endpoint = False
    else:
        # since -90 and 90 generates the same view
        start = offset - (num_modes // 2) * (180 / num_modes)
        end = offset + (num_modes // 2) * (180 / num_modes)
        endpoint = True

    angles = np.linspace(start, end, num_modes, endpoint=endpoint, dtype=int)

    return angles


# Loading and saving functions for videos and images
def load_tensor(filename: str) -> torch.tensor:
    return torch.load(f"{filename}.pt")

def load_images(dir_name: str) -> torch.tensor:
    """Loads all images corresponding to one sample/video

    Args:
        filename (str): original name of video, 
        now the name of the directory containing the images

    Returns: 
        One stacked tensor containing all images    
    """
    if not os.path.exists(dir_name):
        raise FileNotFoundError(dir_name)

    transform = transforms.Compose([transforms.PILToTensor()])
    images = os.listdir(dir_name)
    if len(images) < 1:
        print(f"========================== {dir_name} ==============================")
    image_tensors = []

    # Read images and stack them
    for image in images:
        rgb_image = Image.open(os.path.join(dir_name, image))
        grey_image = rgb_image.convert('L')
        grey_image_tensor = transform(grey_image)
        image_tensors.append(torch.squeeze(grey_image_tensor)) # squeeze to remove channel dimension (which is 1)
        
    images = torch.stack(image_tensors) # format is [#frames, height, width]

    return images


def load_video(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename) 

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Warning: Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray image
        v[count, :, :] = frame # format is [#frames, height, width]

    capture.release()

    return v


def save_video(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    _, height, width = array.shape

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in array:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

        
# Plotting functions
def plot_score(metric, labels, predicted_labels, thresh, output, plot=False):
    fig = plt.figure(figsize=(3, 3))

    if metric == "AUROC":
        plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
        plt.ylabel("True Positive Rate")
        filename = "test_roc.pdf"
        x, y, _ = roc_curve(labels, predicted_labels) 
        score = roc_auc_score(labels, predicted_labels)

    elif metric == "AUPRC":
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        filename = "test_prc.pdf"
        x, y, _ = precision_recall_curve(labels, predicted_labels)
        score = average_precision_score(labels, predicted_labels)

    else:
        raise NotImplementedError

    score = round(score, 2)

    print(f"{metric} for threshold {thresh}: {score}")
    plt.plot(x, y)
    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.tight_layout()
    plt.text(0.5, 0.1, f"AUC: {score}")
    plt.savefig(os.path.join(output, filename))
    plt.close(fig)

    return score


def plot_train_size(sizes, max_size, curriculum, warmup, thresh, factor, \
    init_fraction, epoch_fraction):
    # Set sns style
    sns.set(style="whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3.0})
    ALPHA = 0.7

    # Plot train set size over epochs
    epochs = range(0, len(sizes))
    plot = sns.lineplot(epochs, sizes, drawstyle='steps-pre', alpha=ALPHA)
    # plot.set(xticks=epochs)
    plt.axhline(y=max_size, color='k', linestyle='dotted', label="max_size", alpha=ALPHA)
    plt.xlabel("epoch")
    plt.ylabel("train set size")

    out_dir = os.path.join("plots", "curriculum_learning", f"{curriculum}")
    os.makedirs(out_dir, exist_ok=True)


    if curriculum == "vanilla":
        # plt.axvline(x=, color='r', linestyle='dotted', label="warmup", alpha=ALPHA)

        fig = plot.get_figure()
        fig.savefig(os.path.join(out_dir, f"train_size_CL_{init_fraction}_{epoch_fraction}.png"))
        
    elif curriculum == "self-paced":
        plt.axvline(x=warmup-1, color='r', linestyle='dotted', label="warmup", alpha=ALPHA)

        fig = plot.get_figure()
        fig.savefig(os.path.join(out_dir, f"train_size_SPL_{warmup}_{thresh}_{factor}.png"))

def generate_mask(labels):
    '''
    labels: (batch_size, nb_modes, in_channels)
    '''
    labels = labels.reshape(labels.size(0), -1)
    masks = torch.zeros_like(labels)
    labels_unique = []
    for i in range(labels.size(0)):
        unique, inverse = torch.unique(labels[i], sorted=True, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        masks[i, perm] = 1
        labels_unique.extend(unique)
    labels_unique = torch.tensor(labels_unique)
    labels_unique = labels_unique.to(torch.long)
    return labels_unique, masks

class GaussianNoise(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
    def __call__(self, img):
        noise = torch.randn(img.shape) * self.sigma + self.mean
        img_ = img / 255.0 + noise
        img_ = torch.clip(img_, min=0, max=1)
        img_ = (img_ * 255).to(torch.uint8)
        return img_

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




