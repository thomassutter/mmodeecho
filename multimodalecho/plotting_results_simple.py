import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot(df, dir_results, dataset_name, task):
    print(df.shape)
    # df = df[df["task"] == task]
    print(df.shape)
    models = df.model.unique()
    num_modes = sorted(df.num_modes.unique())
    res = []
    for model in models:
        df_m = df[df["model"] == model]
        for n in sorted(num_modes):
            df_m_n = df_m[df_m["num_modes"] == n]
            mean_auprc = df_m_n["AUPRC"].mean(axis=0)
            std_auprc = df_m_n["AUPRC"].std(axis=0)
            mean_auroc = df_m_n["AUROC"].mean(axis=0)
            std_auroc = df_m_n["AUROC"].std(axis=0)
            # mean_r2 = df_m_n["r2"].mean()
            # std_r2 = df_m_n["r2"].std()
            r_m_n = [
                model,
                int(n),
                mean_auroc,
                std_auroc,
                mean_auprc,
                std_auprc,
                # mean_r2,
                # std_r2,
            ]
            res.append(r_m_n)
    df_avg = pd.DataFrame(
        res,
        columns=[
            "model",
            "num_modes",
            "mean_auroc",
            "std_auroc",
            "mean_auprc",
            "std_auprc",
            # "mean_r2",
            # "std_r2",
        ],
    )
    # df_avg.sort_values(["model", "num_modes"])

    # plot AUPRC
    fn_results_auprc = os.path.join(
        dir_results, "plot_auprc_" + dataset_name + "_" + task + ".png"
    )
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
    x = np.arange(0, len(num_modes))
    for m in df_avg.model.unique():
        sel_m = df_avg["model"] == m
        mean_auprc_v = df_avg[sel_m]["mean_auprc"].values
        std_auprc_v = df_avg[sel_m]["std_auprc"].values
        ax1.plot(num_modes, mean_auprc_v, label=m)
        ax1.fill_between(
            num_modes,
            mean_auprc_v - std_auprc_v,
            mean_auprc_v + std_auprc_v,
            alpha=0.33,
        )
    if dataset_name == "dynamic":
        ax1.hlines(
            0.93,
            num_modes[0],
            num_modes[-1],
            color="purple",
            linestyle="--",
            label="EchoNet",
        )
        ax1.hlines(
            0.76,
            num_modes[0],
            num_modes[-1],
            color="black",
            linestyle="--",
            label="Random",
        )
    ax1.set_ylim([0.7, 1.0])
    ax1.set_xticks(num_modes)
    ax1.set_xlabel("Number of  Modes")
    ax1.set_ylabel("AUPRC")
    ax1.grid()
    # Shrink current axis by 20%
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width * 0.80, box.height])
    plt.draw()
    plt.savefig(fn_results_auprc, format="png")
    plt.close()

    fn_results_auroc = os.path.join(
        dir_results, "plot_auroc_" + dataset_name + "_" + task + ".png"
    )
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig, ax2 = plt.subplots(1, 1, figsize=(5, 4))
    # plot AUROC
    for m in df_avg.model.unique():
        sel_m = df_avg["model"] == m
        mean_auroc_v = df_avg[sel_m]["mean_auroc"].values
        std_auroc_v = df_avg[sel_m]["std_auroc"].values
        ax2.plot(num_modes, mean_auroc_v, label=m)
        ax2.fill_between(
            num_modes,
            mean_auroc_v - std_auroc_v,
            mean_auroc_v + std_auroc_v,
            alpha=0.33,
        )
    if dataset_name == "dynamic":
        ax2.hlines(
            0.96,
            num_modes[0],
            num_modes[-1],
            color="purple",
            linestyle="--",
            label="EchoNet",
        )
        ax2.hlines(
            0.5,
            num_modes[0],
            num_modes[-1],
            color="black",
            linestyle="--",
            label="Random",
        )
    ax2.set_ylim([0.45, 1.0])
    ax2.set_xticks(num_modes)
    ax2.set_xlabel("Number of  Modes")
    ax2.set_ylabel("AUROC")
    ax2.grid()
    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0, box.width * 0.77, box.height])
    plt.draw()
    plt.savefig(fn_results_auroc, format="png")

    ncols_l = 1
    handles, labels = ax2.get_legend_handles_labels()
    # legend = fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.54), loc="lower center")
    legend = fig.legend(
        handles, labels, bbox_to_anchor=(0.5, 1.05), ncol=ncols_l, loc="lower center"
    )
    fn_legend = os.path.join(
        dir_results, "legend_auroc_auprc_ncol_" + str(ncols_l) + ".png"
    )
    export_legend(legend, fn_legend)

    plt.close()
    # # plot r2
    # fn_res_r2 = os.path.join(
    #     dir_results, "plot_r2_" + dataset_name + "_" + task + ".png"
    # )
    # fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    # x = np.arange(0, len(num_modes))
    # for m in df_avg.model.unique():
    #     sel_m = df_avg["model"] == m
    #     mean_r2_v = df_avg[sel_m]["mean_r2"].values
    #     std_r2_v = df_avg[sel_m]["std_r2"].values
    #     ax1.plot(num_modes, mean_r2_v, label=m)
    #     ax1.fill_between(
    #         num_modes, mean_r2_v - std_r2_v, mean_r2_v + std_r2_v, alpha=0.33
    #     )
    # if dataset_name == "dynamic":
    #     ax1.hlines(
    #         0.81,
    #         num_modes[0],
    #         num_modes[-1],
    #         color="purple",
    #         linestyle="--",
    #         label="EchoNet",
    #     )
    # # ax1.hlines(0.76, x[0], x[-1], color="black", linestyle="--", label="Random")
    # # ax1.set_ylim([0.7, 1.0])
    # ax1.set_xticks(num_modes)
    # ax1.set_xlabel("Number of  Modes")
    # ax1.set_ylabel("R2-Score")
    # ax1.grid()
    # # Shrink current axis by 20%
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.54), loc="center right")
    # plt.draw()
    # plt.savefig(fn_res_r2, format="png")
    # plt.close()


if __name__ == "__main__":
    dataset_name = "dynamic"
    task_name = "regression"
    # dir_results = "/usr/scratch/projects/multimodalecho/results"
    # dir_results = "/home/thomas/polybox/PhD/projects/multimodality/multiview_heart_echo/multimodalecho/multimodalecho"
    dir_results = (
        "/home/thomas/polybox/PhD/writing/submissions/2022_neurips_MI_workshop"
    )
    # fn_results = os.path.join(dir_results, "results_runs_" + dataset_name + ".csv")
    fn_results = os.path.join(dir_results, "results_resnet34_lstm.csv")
    df_results = pd.read_csv(fn_results)
    plot(df_results, dir_results, dataset_name, task_name)
