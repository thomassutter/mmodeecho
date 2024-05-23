import os
import yaml
import click

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

@click.command("collect_results")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--plot/--no_plot", default=None)
@click.option("--table/--no_table", default=None)
@click.option("--model_type", type=str, default=None)
@click.option("--pretrained/--random_init", default=None)
@click.option("--classification/--regression", default=None)
@click.option("--thresh", type=int, default=None)
@click.option("--num_epochs", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--step_size", type=int, default=None)
@click.option("--batch_size", type=int, default=None)
@click.option("--axis", type=str, default=None)
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
    plot=None,
    table=None,
    model_type=None,
    pretrained=None,
    classification=None,
    thresh=None,
    num_epochs=None,
    lr=None,
    step_size=None,
    batch_size=None,
    axis=None,
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
            if data_dir is None: data_dir = config["collect_results"]["data_dir"]
            if output is None: output = config["collect_results"]["output"]
            if plot is None: plot = config["collect_results"]["plot"]
            if table is None: table = config["collect_results"]["table"]
            if model_type is None: model_type = config["collect_results"]["model_type"]
            if pretrained is None: pretrained = config["collect_results"]["pretrained"]
            if classification is None: classification = config["collect_results"]["classification"]
            if thresh is None: thresh = config["collect_results"]["thresh"]
            if num_epochs is None: num_epochs = config["collect_results"]["num_epochs"]
            if lr is None: lr = config["collect_results"]["lr"]
            if step_size is None: step_size = config["collect_results"]["step_size"]
            if batch_size is None: batch_size = config["collect_results"]["batch_size"]
            if axis is None: axis = config["collect_results"]["axis"]
            if feature_selection is None: feature_selection = config["collect_results"]["feature_selection"]

            if curriculum is None: curriculum = config["collect_results"]["curriculum"]
            if spl_warmup is None: spl_warmup = config["collect_results"]["spl_warmup"]
            if spl_thresh is None: spl_thresh = config["collect_results"]["spl_thresh"]
            if spl_factor is None: spl_factor = config["collect_results"]["spl_factor"]
            if init_fraction is None: init_fraction = config["collect_results"]["init_fraction"]
            if epoch_fraction is None: epoch_fraction = config["collect_results"]["epoch_fraction"]

            if seed is None: seed = config["collect_results"]["seed"]

    except FileNotFoundError as _:
        pass

    # Collect all summary files in a dataframe
    filenames = list(Path(data_dir).rglob("*.json"))
    results = []
    
    for filename in tqdm(filenames, desc="Loading summary files", unit="files"):
        try:
            summary = pd.read_json(filename, lines=True)
            results.append(summary)
        except ValueError:
            continue

    results = pd.concat(results, ignore_index=True)

    # Combine model and init to one model type (e.g. Conv2d_pretrained)
    results["model"] = results["model"].mask(results["init"].eq("pretrained"), results["model"] + "_pre")
    results = results.drop(["init"], axis=1)

    # Query desired results
    if model_type is not None:
        results.query("model == @model_type", inplace=True)
    if thresh is not None:
        results.query("thresh == @thresh", inplace=True)
    if num_epochs is not None:
        results.query("num_epochs == @num_epochs", inplace=True)
    if lr is not None:
        results.query("lr == @lr", inplace=True)
    if step_size is not None:
        results.query("step_size == @step_size", inplace=True)
    if batch_size is not None:
        results.query("batch_size == @batch_size", inplace=True)
    if axis is not None:
        results.query("axis == @axis", inplace=True)
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
    

    # Print results dataframe
    if table:
        pd.options.display.max_rows = 999
        # pd.options.display.float_format = '{:,.2f}'.format

        to_drop = ["thresh", "task", "input", "width", "axis", "pos_rate", \
                   "runtime_init", "runtime_train", "runtime_test", \
                   "memory_init", "memory_train", "memory_test", \
                   "curriculum", \
                   "spl_warmup", "spl_thresh", "spl_factor", \
                   "init_fraction", "epoch_fraction", \
                   "feature_selection"]
        t = results.drop(to_drop, axis=1)

        print(t.sort_values(["model", "num_modes"]))

    # Plot desired metrics
    if plot:
        os.makedirs("plots", exist_ok=True)

        # Set sns style
        sns.set(style="whitegrid")
        sns.set_context("notebook", rc={"lines.linewidth": 3.0})     

        # Metrics to be plotted
        metrics = ["loss", "AUROC", "AUPRC"]

        # Plots
        ALPHA = 0.7

        for task in ["classification", "regression"]:
            if task == "regression":
                metrics.append("r2")

            for axis in ["default"]: #results["axis"].unique():
                for metric in metrics:
                    plt.clf()
                    data = results.query("task == @task & axis == @axis")
                    plot = sns.relplot(data=data, x="num_modes", y=metric, hue="model",
                        kind="line", facet_kws={"legend_out": True}, style="model", dashes=False, alpha=ALPHA)
                    plot.set(xticks=data["num_modes"])

                    # plot = sns.relplot(data=data, x="num_modes", y=metric, hue="model",
                    #     kind="line", facet_kws={"legend_out": True}, hue_order=hue_order, palette=palette, 
                    #     style="model", style_order=hue_order, dashes=style, alpha=ALPHA)

                    # add baselines for AUPRC
                    if metric == "AUPRC":
                        baseline = results["pos_rate"][0]
                        plt.axhline(y=baseline, color='k', linestyle='dotted', label="baseline", alpha=ALPHA)
                        plt.axhline(y=0.93, color='mediumorchid', linestyle='dotted', label="EchoNet-Dynamic", alpha=ALPHA)
                        # plt.text(9.5, 0.926, "EchoNet-Dynamic", color='mediumorchid')
                        plt.ylim([baseline - 0.02, 1])

                    # add baselines for AUROC
                    if metric == "AUROC":
                        baseline = 0.5
                        plt.axhline(y=baseline, color='k', linestyle='dotted', label="baseline", alpha=ALPHA)
                        plt.axhline(y=0.97, color='mediumorchid', linestyle='dotted', label="EchoNet-Dynamic", alpha=ALPHA)
                        # plt.text(9.5, 0.966, "EchoNet-Dynamic", color='mediumorchid')
                        plt.ylim([baseline - 0.02, 1])

                    fig = plot.fig
                    fig.savefig(os.path.join("plots", f"{task}_{axis}_num_modes_vs_{metric}.png"))
