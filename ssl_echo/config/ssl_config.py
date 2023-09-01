from easydict import EasyDict
import numpy as np
import os
import os.path as osp
import json

class SSLConfig(object):
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
        self.opt["train_batch_size"] = 256
        self.opt["val_batch_size"] = 256
        self.opt["test_batch_size"] = 256
        self.opt["num_workers"] = 4
        self.opt["normalize"] = True  # whether to normalize the output of the encoder
        self.opt["combine"] = "concat" # choose from {concat, mean, lstm}
        self.opt["gamma"] = 0.8 # weight for the classification loss

        self.opt["encoder"] = "resnet18"
        self.opt["pretrained"] = False # whether to use pretrained encoder
        self.opt["feature_dim"] = 2048 # encoder output feature dimension (deprecated)
        self.opt["linear_layers"] = [2048, 128] # projector's hidden layer size and output size
        self.opt["start_epoch"] = 0
        self.opt["epochs"] = 300
        self.opt["eval_epoch_begin"] = 0
        self.opt["eval_interval"] = 1
        self.opt["no_val"] = False
        self.opt["optimizer"] = "Adam"
        self.opt["lr"] = 1.0
        self.opt["lr_scheduler"] = "cos"
        self.opt["lr_step"] = 1
        self.opt["weight_decay"] = 0
        self.opt["momentum"] = 0
        self.opt["nesterov"] = False
        self.opt["temperature"] = 0.01
        self.opt["scale_by_temperature"] = True
        self.opt["warmup_epoch"] = 30
        self.opt["save_model"] = True
        self.opt["resume"] = False
        self.opt["resume_checkpoint"] = ""
        self.opt["experiment"] = ""
        self.opt["checkpoint_dir"] = ""
        self.opt["log_dir"] = ""
        self.opt["log_file"] = "log.txt"
        self.opt["percent_train"] = -1.0
        self.opt["percent_val"] = -1.0
        self.opt["percent_test"] = -1.0
        self.opt["sample_mode_train"] = "random"     # choose from {"sequential", "fixed_random", "random"}
        self.opt["sample_mode_val"] = "fixed_random"
        self.opt["sample_mode_test"] = "fixed_random"

        self.opt["nb_thresh"] = 50  # neightborhood threshold
        self.opt["aug"] = "mix"  # {"mix", None}
        self.opt["use_test"] = False # when set True, use validation set for fine-tuning and test set for evaluation
        self.opt["num_clips"] = None

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
        if self.opt["gamma"] == 0.0:
            self.opt["aug"] = None
        self.opt["experiment"] = "ssl_enc[{}]_warmup[{}]_lr[{}]_clips[{}]_modes[{}]_sd[{}]".format(
            self.opt["encoder"], self.opt["warmup_epoch"], 
            self.opt["lr"], self.opt["num_clips"],
            self.opt["num_modes"], self.opt["seed"]
        )
        self.opt["checkpoint_dir"] = "/cluster/home/yurohu/ssl_echo/experiments/"
        self.opt["checkpoint_dir"] = osp.join(self.opt["checkpoint_dir"], self.opt["experiment"])
        self.opt["log_dir"] = osp.join(self.opt["checkpoint_dir"], "logs")

        if not osp.isdir(self.opt["checkpoint_dir"]):
            os.makedirs(self.opt["checkpoint_dir"])
        if not osp.isdir(self.opt["log_dir"]):
            os.makedirs(self.opt["log_dir"])
    

        
        
        

         
        
