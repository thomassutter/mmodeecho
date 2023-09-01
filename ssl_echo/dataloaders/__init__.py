from torch.utils.data import DataLoader
from dataloaders.datasets import mmodeEcho

def make_data_loader(cfg, **kwargs):
    if not cfg["use_test"]:
        train_set = mmodeEcho.MModeEcho(
            data_dir=cfg["data_dir"], 
            num_modes = cfg["num_modes"],
            sample_mode = cfg["sample_mode_train"],
            nb_thresh = cfg["nb_thresh"],
            aug = cfg["aug"],
            percent = cfg["percent_train"],
            split="TRAIN",
            num_clips=cfg["num_clips"])
        
    val_set = mmodeEcho.MModeEcho(
        data_dir=cfg["data_dir"], 
        num_modes = cfg["num_modes"],
        sample_mode = cfg["sample_mode_val"],
        nb_thresh = cfg["nb_thresh"],
        aug = cfg["aug"],
        percent = cfg["percent_val"],
        split="VAL",
        num_clips=cfg["num_clips"])
    
    if cfg["use_test"]:
        test_set = mmodeEcho.MModeEcho(
            data_dir=cfg["data_dir"], 
            num_modes = cfg["num_modes"],
            sample_mode = cfg["sample_mode_test"],
            nb_thresh = cfg["nb_thresh"],
            aug = cfg["aug"],
            percent = cfg["percent_test"],
            split="TEST",
            num_clips=cfg["num_clips"])
        
    if not cfg["use_test"]:
        train_loader = DataLoader(train_set, batch_size=cfg["train_batch_size"], shuffle=True, drop_last=True, **kwargs)

    val_loader = DataLoader(val_set, batch_size=cfg["val_batch_size"], shuffle=cfg["use_test"], drop_last=True, **kwargs)
    if cfg["use_test"]:
        test_loader = DataLoader(test_set, batch_size=cfg["test_batch_size"], shuffle=False, drop_last=True, **kwargs)
    return (train_loader, val_loader) if not cfg["use_test"] else (val_loader, test_loader)