import os
import yaml
import click

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

@click.command("epoch_plots")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_type", type=str, default=None)
@click.option("--pretrained/--random_init", default=None)
@click.option("--classification/--regression", default=None)
@click.option("--lr", type=float, default=None)
@click.option("--step_size", type=int, default=None)
@click.option("--batch_size", type=int, default=None)
@click.option("--axis", type=str, default=None)
@click.option("--num_modes", type=int, default=None)
@click.option("--feature_selection", type=int, default=None)

# Curriculum learning
@click.option("--curriculum", type=str, default=None)
@click.option("--spl_warmup", type=int, default=None)
@click.option("--spl_thresh", type=int, default=None)
@click.option("--spl_factor", type=float, default=None)
@click.option("--init_fraction", type=float, default=None)
@click.option("--epoch_fraction", type=float, default=None)

@click.option("--seed", type=int, default=None)
def run(
    data_dir=None,
    output=None,
    model_type=None,
    pretrained=None,
    classification=None,
    lr=None,
    step_size=None,
    batch_size=None,
    axis=None,
    num_modes=None,
    feature_selection=None,

    curriculum=None,
    spl_warmup=None,
    spl_thresh=None,
    spl_factor=None,
    init_fraction=None,
    epoch_fraction=None,

    seed=None,
):

    # Load YAML config file and set any missing parameters
    try: 
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            # Set parameters not specified via command line
            if data_dir is None: data_dir = config["epoch_plots"]["data_dir"]
            if output is None: output = config["epoch_plots"]["output"]
            if model_type is None: model_type = config["epoch_plots"]["model_type"]
            if pretrained is None: pretrained = config["epoch_plots"]["pretrained"]
            if classification is None: classification = config["epoch_plots"]["classification"]
            if lr is None: lr = config["epoch_plots"]["lr"]
            if step_size is None: step_size = config["epoch_plots"]["step_size"]
            if batch_size is None: batch_size = config["epoch_plots"]["batch_size"]
            if axis is None: axis = config["epoch_plots"]["axis"]
            if num_modes is None: num_modes = config["epoch_plots"]["num_modes"]
            if feature_selection is None: feature_selection = config["epoch_plots"]["feature_selection"]

            if curriculum is None: curriculum = config["epoch_plots"]["curriculum"]
            if spl_warmup is None: spl_warmup = config["epoch_plots"]["spl_warmup"]
            if spl_thresh is None: spl_thresh = config["epoch_plots"]["spl_thresh"]
            if spl_factor is None: spl_factor = config["epoch_plots"]["spl_factor"]
            if init_fraction is None: init_fraction = config["epoch_plots"]["init_fraction"]
            if epoch_fraction is None: epoch_fraction = config["epoch_plots"]["epoch_fraction"]

            if seed is None: seed = config["epoch_plots"]["seed"]

    except FileNotFoundError as _:
        pass

    # Collect all relevant results
    results = []

    for root, subdir, files in os.walk(data_dir):
        if not subdir:
            summary = pd.read_json(os.path.join(root, "summary.json"), lines=True)
            log = pd.read_csv(os.path.join(root, "log.csv"))
            results.append(pd.concat([summary, log], axis=1).ffill())

    results = pd.concat(results, ignore_index=True)

    # Query desired results
    if model_type is not None:
        results = results[results["model"].str.contains(model_type)]
    if lr is not None:
        results.query("lr == @lr", inplace=True)
    if step_size is not None:
        results.query("step_size == @step_size", inplace=True)
    if batch_size is not None:
        results.query("batch_size == @batch_size", inplace=True)
    if axis is not None:
        results.query("axis == @axis", inplace=True)
    if num_modes is not None:
        results.query("num_modes == @num_modes", inplace=True)
    if feature_selection is not None:
        results.query("feature_selection == @feature_selection", inplace=True)
    if curriculum is not None:
        results.query("curriculum == @curriculum", inplace=True)
    if spl_warmup is not None:
        results.query("spl_warmup == @spl_warmup", inplace=True)
    if spl_thresh is not None:
        results.query("spl_thresh == @spl_thresh", inplace=True)
    if spl_factor is not None:
        results.query("spl_factor == @spl_factor", inplace=True)
    if init_fraction is not None:
        results.query("init_fraction == @init_fraction", inplace=True)
    if epoch_fraction is not None:
        results.query("epoch_fraction == @epoch_fraction", inplace=True)

    # Check whether curriculum learning results are available
    if curriculum is not None:
        curr = curriculum
    else:
        curr = results["curriculum"].unique()
        if np.isnan(curr[0]):
            curr = None

    # Combine model and init to one model type (e.g. Conv2d_pretrained)
    results["model"] = results["model"].mask(results["init"].eq("pretrained"), results["model"] + "_pre")
    results = results.drop(["init"], axis=1)

    if curr is not None and "vanilla" in curr:
        # merge CL params for nicer display later
        fractions = results.agg(lambda x: f"IF{x['init_fraction']:1.1f}_EF{x['epoch_fraction']:1.1f}", axis=1)
        results = results.drop(["init_fraction"], axis=1)
        results = results.drop(["epoch_fraction"], axis=1)
        results.insert(0, "fractions", fractions)

    # Divide results into train and validation results
    results_train = results.query("phase == 'TRAIN'")
    if curr is not None:
        results_train = results_train[["model", "task", "axis", "epoch", "epoch_loss", \
            "curriculum", "train_size", "spl_warmup", "fractions"]]
        results_train["train_size"] = pd.to_numeric(results_train["train_size"])
    else:
        results_train = results_train[["model", "task", "axis", "epoch", "epoch_loss"]]

    results_val = results.query("phase == 'VAL'")
    if curr is not None:
        results_val = results_val[["model", "task", "axis", "epoch", "auroc", "auprc", "epoch_loss", \
            "curriculum", "spl_warmup", "fractions"]]
        out_path_train = os.path.join(f"{output}_CL", "TRAIN")
        out_path_val = os.path.join(f"{output}_CL", "VAL")
    else:
        results_val = results_val[["model", "task", "axis", "epoch", "auroc", "auprc", "epoch_loss"]]
        out_path_train = os.path.join(output, "TRAIN")
        out_path_val = os.path.join(output, "VAL")

    results_val["auroc"] = pd.to_numeric(results_val["auroc"])
    results_val["auprc"] = pd.to_numeric(results_val["auprc"])
    
    os.makedirs(out_path_train, exist_ok=True)
    os.makedirs(out_path_val, exist_ok=True)

    # Set sns style
    sns.set(style="whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3.0})

    # Metrics to be plotted
    metrics_train = ["epoch_loss"]
    if curr is not None:
        metrics_train.append("train_size")

    metrics_val = ["epoch_loss", "auroc", "auprc"]

    # Plots
    ALPHA = 0.7
    if curr is None:
        for task in ["regression"]:
            for axis in ["default"]: #results["axis"].unique():
                for metric in metrics_train:
                    plt.clf()
                    data = results_train.query("task == @task & axis == @axis")
                    plot = sns.relplot(data=data, x="epoch", y=metric, hue="model",
                        kind="line", facet_kws={"legend_out": True}, style="model", 
                        dashes=False, alpha=ALPHA, ci="sd")

                    if metric == "epoch_loss":
                        plot.set(ylim=(0, 100))

                    fig = plot.fig
                    fig.savefig(os.path.join(out_path_train, f"{task}_{axis}_epoch_vs_{metric}.png"))

                for metric in metrics_val:
                    plt.clf()
                    data = results_val.query("task == @task & axis == @axis")
                    plot = sns.relplot(data=data, x="epoch", y=metric, hue="model",
                        kind="line", facet_kws={"legend_out": True}, style="model", 
                        dashes=False, alpha=ALPHA, ci="sd")

                    if metric == "epoch_loss":
                        plot.set(ylim=(50, 150))

                    fig = plot.fig
                    fig.savefig(os.path.join(out_path_val, f"{task}_{axis}_epoch_vs_{metric}.png"))


    else: # curriculum
        for task in ["regression"]:
            for axis in ["default"]: #results["axis"].unique():
                for curriculum in curr:
                    if curriculum == "vanilla":
                        hue = "fractions"
                    elif curriculum == "self-paced":
                        hue = "spl_warmup"

                    for metric in metrics_train:
                        plt.clf()
                        data = results_train.query("task == @task & axis == @axis & curriculum == @curriculum")
                        # pd.options.display.max_rows = 999
                        # print(data)
                        plot = sns.relplot(data=data, x="epoch", y=metric, hue=hue,
                            kind="line", facet_kws={"legend_out": True}, style=hue, 
                            dashes=False, alpha=ALPHA, ci="sd")

                        # if metric == "epoch_loss":
                        #     plot.set(ylim=(0, 100))

                        fig = plot.fig
                        fig.savefig(os.path.join(out_path_train, f"{task}_{axis}_{curriculum}_epoch_vs_{metric}.png"))

                    for metric in metrics_val:
                        plt.clf()
                        data = results_val.query("task == @task & axis == @axis & curriculum == @curriculum")
                        plot = sns.relplot(data=data, x="epoch", y=metric, hue=hue,
                            kind="line", facet_kws={"legend_out": True}, style=hue, 
                            dashes=False, alpha=ALPHA, ci="sd")

                        # if metric == "epoch_loss":
                        #     plot.set(ylim=(50, 150))

                        fig = plot.fig
                        fig.savefig(os.path.join(out_path_val, f"{task}_{axis}_{curriculum}_epoch_vs_{metric}.png"))
