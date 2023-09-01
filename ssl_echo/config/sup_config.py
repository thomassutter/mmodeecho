from easydict import EasyDict
import numpy as np
import os
import os.path as osp
import json

class SupConfig(object):
    def __init__(self, fn=None):
        self.opt = EasyDict({})
        self.set_default_opts()
        if fn:
            self.load_from_json(fn)

    def set_default_opts(self):
        self.fn = None
        self.opt["seed"] = 0
        self.opt["data_dir"] = "/cluster/work/vogtlab/Projects/EchoNet-Mmode"  # data directory
        self.opt["data_type"] = "tensors"  # choose from {images, videos, tensors}, readin data type
        self.opt["frames"] = False  # only valid when data_type=videos. True: use individual frames from each video; False: M-mode
        self.opt["num_modes"] = 10  # number of modes per patient
        self.opt["in_channels"] = 1  # number of input channels
        self.opt["axis"] = "default"  # deprecated
        self.opt["window_width"] = 32  # fixed number of frames per video or clip, discarding shorter ones and truncating longer ones
        self.opt["train_batch_size"] = 64
        self.opt["val_batch_size"] = 64
        self.opt["test_batch_size"] = 64
        self.opt["num_workers"] = 4
        self.opt["encoder"] = "resnet18"
        self.opt["freeze"] = False
        self.opt["feature_dim"] = 2048
        self.opt["linear_layers"] = [2048, 128]
        self.opt["hidden_size"] = 256  # deprecated
        self.opt["saved_pretrained_model"] = None
        self.opt["start_epoch"] = 0
        self.opt["epochs"] = 100
        self.opt["eval_epoch_begin"] = 0
        self.opt["eval_interval"] = 1
        self.opt["no_val"] = False
        self.opt["optimizer"] = "Adam"
        self.opt["lr"] = 1e-3
        self.opt["lr_scheduler"] = "poly"
        self.opt["lr_step"] = 15
        self.opt["weight_decay"] = 0
        self.opt["momentum"] = 0
        self.opt["nesterov"] = False
        self.opt["normalize"] = False
        self.opt["pretrained"] = False
        self.opt["combine"] = "concat"  # {"concat", "avg", "lstm", None}
        self.opt["save_model"] = True
        self.opt["resume"] = False
        self.opt["resume_checkpoint"] = ""
        self.opt["start_epoch"] = 0
        self.opt["ef_threshold"] = 50 # threshold for calculating AUROC & AUPRC
        self.opt["experiment"] = ""
        self.opt["checkpoint_dir"] = ""
        self.opt["log_dir"] = ""
        self.opt["log_file"] = "log.txt"
        self.opt["ckpt"] = "best"  # choose from ["best", "last"], deprecated

        self.opt["percent_train"] = -1.0
        self.opt["percent_val"] = -1.0
        self.opt["percent_test"] = -1.0

        # {"sequential", "fixed_random", "random"}
        self.opt["sample_mode_train"] = "random"     
        self.opt["sample_mode_val"] = "fixed_random"
        self.opt["sample_mode_test"] = "fixed_random"

        self.opt["nb_thresh"] = 50
        self.opt["clf"] = "linear"  # {"linear", "mlp"}
        self.opt["aug"] = None
        self.opt["use_test"] = False
        self.opt["num_clips"] = 1

    def __getitem__(self, key):
        return self.opt[key]
    
    def load_from_json(self, fn):
        self.fn = fn
        with open(fn, "r") as f:
            jdata = json.load(f)
        for key in jdata.keys():
            self.opt[key] = jdata[key]
        print("load opt from: {:s}".format(fn))
    
    def write_to_json(self, fn=None):
        jdata = json.dumps(self.opt, sort_keys=True, indent=4)
        if fn is None:
            fn = osp.join(self.opt["log_dir"], "config.json")
        with open(fn, "w") as f:
            f.write(jdata)
        print("write opt to {:s}".format(fn))
    
    def set_environmental_variables(self):
        if self.opt["use_test"] == False:
            percent = self.opt["percent_train"]
        else:
            percent = self.opt["percent_val"]
        if self.opt["combine"] is None:
            self.opt["in_channels"] = self.opt["num_modes"]
        self.opt["experiment"] = "sup_percent[{}]_comb[{}]_freeze[{}]_lr[{}]_clips[{}]_modes[{}]_sd[{}]".format(
            percent, self.opt["combine"],
            self.opt["freeze"], self.opt["lr"], 
            self.opt["num_clips"], self.opt["num_modes"], 
            self.opt["seed"]
        )
        if self.opt["saved_pretrained_model"] is None:
            root_dir = "/cluster/home/yurohu/ssl_echo/experiments/e2e/encoder[{}]_pretrained[{}]".format(
                self.opt["encoder"],
                self.opt["pretrained"]
            )
        else:
            root_dir = osp.split(self.opt["saved_pretrained_model"])[0]
        self.opt["checkpoint_dir"] = osp.join(root_dir, self.opt["experiment"])
        self.opt["log_dir"] = osp.join(self.opt["checkpoint_dir"], "logs")

        if not osp.isdir(self.opt["checkpoint_dir"]):
            os.makedirs(self.opt["checkpoint_dir"])
        if not osp.isdir(self.opt["log_dir"]):
            os.makedirs(self.opt["log_dir"])
    

        
        
        

         
        
