import os.path as osp
import shutil
import torch
import json
import sys

class Saver(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_dir = cfg["checkpoint_dir"]
        self.log_dir = cfg["log_dir"]

    def save_checkpoint(self, state, is_best, filename="checkpoint.pth.tar", save_model=True):
        """Saves checkpoint to disk"""
        filename = osp.join(self.checkpoint_dir, filename)
        if save_model: torch.save(state, filename)
        if is_best:
            best_loss = state["best_loss"]
            with open(osp.join(self.log_dir, "best_loss.txt"), "w") as f:
                json.dump(best_loss, f)
            if "best_lna" in state:
                best_lna = state["best_lna"]
                with open(osp.join(self.log_dir, "best_lna.txt"), "w") as f:
                    json.dump(best_lna, f)
            if "best_lnd" in state:
                best_lnd = state["best_lnd"]
                with open(osp.join(self.log_dir, "best_lnd.txt"), "w") as f:
                    json.dump(best_lnd, f)
            if "best_mae" in state:
                best_mae = state["best_mae"]
                with open(osp.join(self.log_dir, "best_mae.txt"), "w") as f:
                    json.dump(best_mae, f)
            if "best_auroc" in state:
                best_auroc = state["best_auroc"]
                with open(osp.join(self.log_dir, "best_auroc.txt"), "w") as f:
                    json.dump(best_auroc, f) 
            if "best_auprc" in state:
                best_auprc = state["best_auprc"]
                with open(osp.join(self.log_dir, "best_auprc.txt"), "w") as f:
                    json.dump(best_auprc, f) 
            if save_model: shutil.copyfile(filename, osp.join(self.checkpoint_dir, "model_best.pth.tar"))


class Logger(object):
    def __init__(self, cfg):
        self.terminal = sys.stdout
        filename = osp.join(cfg["log_dir"], cfg["log_file"])
        self.log = open(filename, "w", buffering=1)

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass