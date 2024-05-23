import os
import json
import yaml
import time
import click

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import random

import multimodalecho
from multimodalecho.utils.models import get_model_layout, get_model_layout2d, ResNet_1d, ResNet_lstm, ResNet_lstm2d
from multimodalecho.utils.losses import SPLLoss_regression, CLLoss_regression

@click.command("run_model")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_type", type=str, default=None)
@click.option("--pretrained/--random_init", default=None)
@click.option("--classification/--regression", default=None)
@click.option("--datatype", type=str, default=None)
@click.option("--thresh", type=int, default=None)
@click.option("--num_epochs", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--step_size", type=int, default=None)
@click.option("--batch_size", type=int, default=None)

# M-mode specific options
@click.option("--frames/--Mmode", default=None)
@click.option("--num_modes", type=int, default=None)
@click.option("--axis", type=str, default=None)
@click.option("--width", type=int, default=None)

@click.option("--feature_selection", type=int, default=None)

# Curriculum learning
@click.option("--curriculum", type=str, default=None)
@click.option("--spl_warmup", type=int, default=None)
@click.option("--spl_thresh", type=int, default=None)
@click.option("--spl_factor", type=float, default=None)
@click.option("--spl_quantile", type=float, default=None)
@click.option("--init_fraction", type=float, default=None)
@click.option("--epoch_fraction", type=float, default=None)

@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--tensorboard", type=bool, default=None)
def run(
    data_dir=None,
    output=None,
    model_type=None,
    pretrained=None,
    classification=None,
    datatype=None,
    thresh=None,
    num_epochs=None,
    lr=None,
    step_size=None,
    batch_size=None,

    frames=None,
    axis=None,
    num_modes=None,
    width=None,

    feature_selection=None,

    curriculum=None,
    spl_warmup=None,
    spl_thresh=None,
    spl_factor=None,
    spl_quantile=None,
    init_fraction=None,
    epoch_fraction=None,

    device=None,
    seed=None,
    tensorboard=None
):
    # Load YAML config file and set any missing parameters
    try: 
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            # Set parameters not specified via command line
            if data_dir is None: data_dir = config["run_model"]["data_dir"]
            if output is None: output = config["run_model"]["output"]
            if model_type is None: model_type = config["run_model"]["model_type"]
            if pretrained is None: pretrained = config["run_model"]["pretrained"]
            if classification is None: classification = config["run_model"]["classification"]
            if datatype is None: datatype = config["run_model"]["datatype"]
            if thresh is None: thresh = config["run_model"]["thresh"]
            if num_epochs is None: num_epochs = config["run_model"]["num_epochs"]
            if lr is None: lr = config["run_model"]["lr"]
            if step_size is None: step_size = config["run_model"]["step_size"]
            if batch_size is None: batch_size = config["run_model"]["batch_size"]

            if frames is None: frames = config["run_model"]["frames"]
            if axis is None: axis = config["run_model"]["axis"]
            if num_modes is None: num_modes = config["run_model"]["num_modes"]
            if width is None: width = config["run_model"]["width"]

            if feature_selection is None: feature_selection = config["run_model"]["feature_selection"]

            if curriculum is None: curriculum = config["run_model"]["curriculum"]
            if spl_warmup is None: spl_warmup = config["run_model"]["spl_warmup"]
            if spl_thresh is None: spl_thresh = config["run_model"]["spl_thresh"]
            if spl_factor is None: spl_factor = config["run_model"]["spl_factor"]
            if spl_quantile is None: spl_quantile = config["epoch_plots"]["spl_quantile"]
            if init_fraction is None: init_fraction = config["run_model"]["init_fraction"]
            if epoch_fraction is None: epoch_fraction = config["run_model"]["epoch_fraction"]

            if device is None: device = config["run_model"]["device"]
            if seed is None: seed = config["run_model"]["seed"]
            if tensorboard is None: tensorboard = config["run_model"]["tensorboard"]

    except FileNotFoundError as _:
        pass

    # Depending on environment, None is sometimes read in as string
    if feature_selection == "None":
        feature_selection = None
    if curriculum == "None":
        curriculum = None

    # Set default output directory if not set already
    if output is None:
        output = os.path.join("output",
            f"{'classification' if classification else 'regression'}",
            f"{model_type}_{'pretrained' if pretrained else 'random_init'}",
            f"{f'{curriculum}' if curriculum is not None else 'standard'}",
            f"{f'vanilla_{epoch_fraction}_{init_fraction}' if curriculum == 'vanilla' else ''}"
            f"{f'SPL_{spl_warmup}_{spl_thresh}_{spl_factor}_{spl_quantile}' if curriculum == 'self-paced' else ''}",
            f"{f'frames_{num_modes}' if frames else f'Mmode_{width}_{axis}_{num_modes}'}"
            f"{f'_fs{feature_selection}' if feature_selection is not None else ''}",
            f"lr{lr:.0e}_ss{step_size}_bs{batch_size}",
            f"seed{seed}"
        )

    # Seed RNGs for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
    # Check whether experiment has already been conducted
    # If summary file is available, check number of epochs
    try:
        with open(os.path.join(output, "summary.json"), "r") as s:
            summary = json.load(s)
            if summary["num_epochs"] >= num_epochs:
                print("Experiment already conducted, skipping execution.")
                return

    except (FileNotFoundError, KeyError) as _:
        pass

    if tensorboard:
        tensorboard_dir = os.path.join("tensorboard",
            f"{'classification' if classification else 'regression'}",
            f"{num_modes}",
            f"{model_type}_{'pretrained' if pretrained else 'random_init'}",
            f"{f'{curriculum}' if curriculum is not None else 'standard'}",
            f"{f'vanilla_{epoch_fraction}_{init_fraction}' if curriculum == 'vanilla' else ''}"
            f"{f'SPL_{spl_warmup}_{spl_thresh}_{spl_factor}_{spl_quantile}' if curriculum == 'self-paced' else ''}",
            f"{f'frames_{num_modes}' if frames else f'Mmode_{width}_{axis}_{num_modes}'}"
            f"{f'_fs{feature_selection}' if feature_selection is not None else ''}",
            f"lr{lr:.0e}_ss{step_size}_bs{batch_size}",
            f"seed{seed}"
        )

    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set number of input channels
    if feature_selection is not None:
        in_channels = num_modes - 1
    else:
        in_channels = num_modes

    # Set up model
    if "2d_lstm" in model_type:
        # no pretrained version of this model is available
        if pretrained:
            print(f"No pretrained version of the {model_type} model is available.")
            raise NotImplementedError
            
        block_type, num_block_list = get_model_layout2d(model_type)
        model = ResNet_lstm2d(block_type, num_block_list, in_channels, num_filters=64)

    elif "2d" in model_type:
        # pretrained models cannot be loaded on leomed, therefore they are pre-loaded
        if pretrained == False:
            try: # some models can be pre-loaded even if pretrained=False
                model_filename = f"../pretrainedModels/hub/checkpoints/{model_type}.pt"
                model = torch.load(model_filename)
            except IOError:
                print("Preloaded model file does not exist.")
        else: # pretrained == True
            model_filename = f"../pretrainedModels/hub/checkpoints/{model_type}_pretrained.pt"
            model = torch.load(model_filename)

        # adapt model to accept inputs of format [in_channels, height, width]
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    elif "1d_lstm" in model_type:
        # no pretrained version of this model is available
        if pretrained:
            print(f"No pretrained version of the {model_type} model is available.")
            raise NotImplementedError
            
        block_type, num_block_list = get_model_layout(model_type)
        model = ResNet_lstm(block_type, num_block_list, in_channels, num_filters=64, kernel_height=112)

    elif "1d" in model_type:
        # no pretrained version of this model is available
        if pretrained:
            print(f"No pretrained version of the {model_type} model is available.")
            raise NotImplementedError
            
        block_type, num_block_list = get_model_layout(model_type)
        model = ResNet_1d(block_type, num_block_list, in_channels, num_filters=64, kernel_height=112)


    else:
        print("The model type you requested is not available.")
        raise NotImplementedError

    if classification:
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    else: # regression
        model.fc = torch.nn.Linear(model.fc.in_features, 1)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    if classification:
        criterion = nn.CrossEntropyLoss()
    else: # regression
        criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    if tensorboard:
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None

    trained_model, runtimes, memory = train_model(data_dir, output, datatype, classification, model, criterion, optimizer, \
                                scheduler, batch_size, thresh, frames, num_modes, in_channels, axis, width, device, \
                                num_epochs, writer, feature_selection, \
                                curriculum, spl_warmup, spl_thresh, spl_factor, spl_quantile, init_fraction, epoch_fraction)

    test_start_time = time.time()
    if tensorboard:
        writer.flush()  
        writer.close()                              
    trained_model.eval()

    # Reset max memory allocated on GPU
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_max_memory_allocated(i)

    # Load test data for final model evaluation
    test_dataset = multimodalecho.datasets.Multimodalecho(data_dir, datatype, frames, num_modes, \
        in_channels, axis, width, feature_selection, split="TEST")
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    true_labels = torch.empty(len(test_data), dtype=torch.long)
    if classification:
        predicted_labels = torch.empty((len(test_data), 2))
    else: # regression
        predicted_labels = torch.empty(len(test_data))

    # writer.add_graph(model)

    # Run the trained model on the test set
    index = 0
    with torch.no_grad():
        with tqdm(test_data, desc=f"TEST", unit="batch") as test_d:
            for inputs, labels in test_d:
                inputs = inputs.to(device)            
                labels = labels.to(device)

                predicted_labels[index] = trained_model(inputs)
                true_labels[index] = labels
                index += 1

        # Compute and plot metrics while saving to summary.json
        summary = {
            "thresh": thresh,
            "model" : model_type,
            "init": f"{'pretrained' if pretrained else 'random_init'}",
            "task": f"{'classification' if classification else 'regression'}",
            "input": f"{'frames' if frames else 'Mmode'}",
            "width": width,
            "axis": axis,
            "num_modes": num_modes,
            "num_epochs": num_epochs,
            "lr": lr,
            "step_size": step_size,
            "batch_size": batch_size,
            "feature_selection": feature_selection,
            "curriculum": curriculum,
            "spl_warmup": spl_warmup,
            "spl_thresh": spl_thresh,
            "spl_factor": spl_factor,
            "spl_quantile": spl_quantile,
            "init_fraction": init_fraction,
            "epoch_fraction": epoch_fraction,
            "seed": seed
        }

        memory["test"] = sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count()))

        if not classification: # regression
            summary["r2"] = r2_score(true_labels, predicted_labels)
            loss = criterion(predicted_labels, true_labels)

        # binarise labels to {0, 1} for remaining calculations
        # for comparability with EchoNet, the positive class corresponds to healthy patients (EF > thresh)
        true_labels = (true_labels > thresh).to(torch.long) 

        if classification: # class label targets more efficient for CELoss
            loss = criterion(predicted_labels, true_labels)
        
        summary["loss"] = loss.item()

        if classification:
            predicted_labels = F.softmax(predicted_labels[:, 1], dim=0) # compute probabilities for positive class

        summary["AUROC"] = multimodalecho.utils.plot_score("AUROC", true_labels, predicted_labels, thresh=thresh, output=output)
        summary["AUPRC"] = multimodalecho.utils.plot_score("AUPRC", true_labels, predicted_labels, thresh=thresh, output=output)
        summary["pos_rate"] = len(true_labels[true_labels == 1]) / len(true_labels) # AUPRC baseline

        # store runtimes in minutes
        runtimes["test"] = (time.time() - test_start_time) // 60
        summary["runtime_init"] = runtimes["init"]
        summary["runtime_train"] = runtimes["train"]
        summary["runtime_test"] = runtimes["test"]

        # store memory usage
        summary["memory_init"] = memory["init"]
        summary["memory_train"] = memory["train"]
        summary["memory_test"] = memory["test"]

        with open(os.path.join(output, "summary.json"), "w") as s:
            json.dump(summary, s)


def train_model(data_dir, output, datatype, classification, model, criterion, optimizer, \
                scheduler, batch_size, thresh, frames, num_modes, in_channels, axis, width, \
                device, num_epochs, writer, feature_selection, \
                curriculum, spl_warmup, spl_thresh, spl_factor, spl_quantile, init_fraction, epoch_fraction): 

    init_start_time = time.time()
    runtimes = {"init": 0, "train": 0}
    memory = {"init": 0, "train": 0}

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 1
        best_loss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"), map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            runtimes["init"] = checkpoint["runtime_init"]
            runtimes["train"] = checkpoint["runtime_train"]
            memory["init"] = checkpoint["memory_init"]
            memory["train"] = checkpoint["memory_train"]
            if curriculum == "self-paced":
                spl_thresh = checkpoint["spl_thresh"]

        except FileNotFoundError:
            pass

        # Exit training if desired number of epochs has already been reached
        if epoch_resume > num_epochs:
            return model, runtimes, memory

        # Add header line to csv only if starting new run
        if epoch_resume == 1:
            f.write("epoch,phase,epoch_loss,auroc,auprc,epoch_cuda_mem,train_size\n")

        # Reset max memory allocated on GPU
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(i)

        # Initialize datasets
        dataset = {}
        for phase in ['TRAIN', 'VAL']:
            dataset[phase] = multimodalecho.datasets.Multimodalecho(data_dir, datatype, frames, \
                num_modes, in_channels, axis, width, feature_selection, split=phase)

        # Initialize special loss for curriculum learning
        if curriculum == "self-paced":
            train_criterion = SPLLoss_regression(threshold=spl_thresh, growing_factor=spl_factor, \
                quantile=spl_quantile)
        elif curriculum == "vanilla":
            train_criterion = CLLoss_regression(batch_size=batch_size, num_epochs=num_epochs, \
                init_fraction=init_fraction, epoch_fraction=epoch_fraction)

        runtimes["init"] = (time.time() - init_start_time) // 60
        print(f"Initialisation of TRAIN and VAL took {runtimes['init']} minutes.")

        memory["init"] = sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count()))

        # Run training and validation epochs
        for epoch in range(epoch_resume, num_epochs+1):     
            epoch_start_time = time.time()
            for phase in ['TRAIN', 'VAL']: 
                # Set up data and loss
                data = torch.utils.data.DataLoader(dataset[phase], num_workers=16, batch_size=batch_size, shuffle=True, \
                    pin_memory=True, drop_last=True)
                model.train(phase == 'TRAIN')
    
                running_loss = 0.0 

                # Reset max memory allocated on GPU
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)

                # keep track of metrics on validation set during training
                if phase == 'VAL':
                    index = 0
                    size = len(dataset["VAL"]) - len(dataset["VAL"]) % batch_size # last incomplete batch is dropped
                    true_labels = torch.empty(size)
                    predicted_labels = torch.empty(size)
    
                # Iterate over data
                with tqdm(data, desc=f"{phase} epoch {epoch}", unit="batch") as d:
                    for inputs, labels in d: 
                        inputs = inputs.to(device)

                        if classification:
                            # binarise labels to {0, 1}
                            # for comparability with EchoNet, the positive class corresponds to healthy patients (EF > 50%)
                            labels = (labels > thresh).to(torch.long)
                            labels = torch.squeeze(labels)
                        labels = labels.to(device)
        
                        # zero the parameter gradients 
                        optimizer.zero_grad(set_to_none=True) 
        
                        # forward 
                        # track history only if in train 
                        with torch.set_grad_enabled(phase == 'TRAIN'): 
                            outputs = model(inputs) 
                            if classification:
                                loss = criterion(outputs, labels) 
                            else: # regression
                                if curriculum == "vanilla" and phase == 'TRAIN':
                                    loss = train_criterion(epoch, outputs.view(-1), labels)
                                elif curriculum == "self-paced" and epoch > spl_warmup and phase == 'TRAIN':
                                    loss = train_criterion(outputs.view(-1), labels)
                                else:
                                    loss = criterion(outputs.view(-1), labels)
        
                            # backward and optimize only if in train
                            if phase == 'TRAIN': 
                                loss.backward() 
                                optimizer.step() 
                            else: # VAL
                                if classification:
                                    true_labels[index:index+batch_size] = labels
                                    outputs = F.softmax(outputs[:, 1], dim=0)
                                    predicted_labels[index:index+batch_size] = outputs
                                else: # regression
                                    labels = (labels > thresh).to(torch.long)
                                    true_labels[index:index+batch_size] = labels
                                    predicted_labels[index:index+batch_size] = outputs.view(-1)
                                
                                index += batch_size
        
                        running_loss += loss.item()
                
                # take scheduler step only if in train
                if phase == 'TRAIN': 
                    scheduler.step()
                    auroc = None
                    auprc = None
                    if curriculum is not None:
                        train_size = train_criterion.get_train_size()
                    else: 
                        train_size = None
                else: # VAL: compute metrics only if in validation
                    auroc = roc_auc_score(true_labels, predicted_labels)
                    auprc = average_precision_score(true_labels, predicted_labels)
                    train_size = None

                epoch_loss = running_loss / len(data) # divide by number of batches

                epoch_cuda_mem = sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count()))
                memory["train"] = max(memory["train"], epoch_cuda_mem)

                if writer is not None:
                    writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)

                f.write(f"{epoch},{phase},{epoch_loss},{auroc},{auprc},{epoch_cuda_mem},{train_size}\n")
                f.flush

            if writer is not None:
                writer.add_scalar("AUROC/VAL", auroc, epoch)
                writer.add_scalar("AUPRC/VAL", auprc, epoch)


            # SPL: increase threshold to allow harder samples in next epoch
            if curriculum == "self-paced" and epoch > spl_warmup:
                train_criterion.increase_threshold()

            runtimes["train"] += time.time() - epoch_start_time

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'epoch_loss': epoch_loss,
                'opt_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'runtime_init': runtimes["init"],
                'runtime_train': runtimes["train"],
                'memory_init': memory["init"],
                'memory_train': memory["train"]
            }
            if curriculum == "self-paced" and epoch > spl_warmup:
                save["spl_thresh"] = train_criterion.get_threshold()

            torch.save(save, os.path.join(output, "checkpoint.pt"))

            # Deep copy the model if validation loss has decreased
            if phase == "VAL" and epoch_loss < best_loss: 
                torch.save(save, os.path.join(output, "best.pt"))
                best_loss = epoch_loss 

        runtimes["train"] = runtimes["train"] // 60

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    return model, runtimes, memory
