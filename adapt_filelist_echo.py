import os
import pandas as pd


def adapt_file_list(dataset, dir_data):
    if dataset == "dynamic":
        fn = "FileList.csv"
        mmode_folder = "Mmodes_50"
        fn_out_filelist = os.path.join(dir_data, "FileList_Dynamic_new.csv")
    elif dataset == "lvh":
        fn = "FileList_LVH.csv"
        mmode_folder = "Mmodes_LVH_50"
        fn_out_filelist = os.path.join(dir_data, "FileList_LVH_new.csv")
    else:
        print("unknown dataset - return")
        return
    fn_filelist = os.path.join(dir_data, fn)
    df_fn = pd.read_csv(fn_filelist)
    df_fn["dataset"] = dataset
    df_fn["mmode_folder"] = mmode_folder
    df_fn.to_csv(fn_out_filelist)


if __name__ == "__main__":
    dataset = "lvh"
    dir_data = "DIR_DATA_BASE"
    adapt_file_list(dataset, dir_data)
