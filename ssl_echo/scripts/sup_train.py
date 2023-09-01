import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import argparse
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from models.regressor import Regressor
from dataloaders import make_data_loader
from utils import EarlyStopper
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from config.sup_config import SupConfig

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.saver = Saver(cfg)
        self.summary = TensorboardSummary(cfg["log_dir"])

        kwargs = {"num_workers": cfg["num_workers"], "pin_memory": True}
        self.train_dataloader, self.val_dataloader = make_data_loader(cfg, **kwargs)

        # self.early_stopper = EarlyStopper(patience=10, min_delta=10)

        model = Regressor(encoder=cfg["encoder"], 
                          linear_layers=cfg["linear_layers"],
                          saved_pretrained_model=cfg["saved_pretrained_model"],
                          in_channels=cfg["in_channels"],
                          modes=cfg["num_modes"],
                          combine=cfg["combine"],
                          clf=cfg["clf"],
                          freeze=cfg["freeze"],
                          normalize=cfg["normalize"],
                          pretrained=cfg["pretrained"]
                          )
        
        if cfg["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                        lr=cfg["lr"], momentum=cfg["momentum"],
                                        weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
        elif cfg["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                        lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=True)
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        self.model = model
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.criterion = torch.nn.MSELoss()
        self.metric = torch.nn.L1Loss()

        self.scheduler = LR_Scheduler(mode=cfg["lr_scheduler"], base_lr=cfg["lr"],
                                num_epochs=cfg["epochs"], iters_per_epoch=len(
                                    self.train_dataloader),
                                lr_step=cfg["lr_step"])

        if cfg["resume"] == True:
            if not os.path.isfile(cfg["resume_checkpoint"]):
                raise RuntimeError("=> no checkpoint found at {:s}" .format(
                    cfg["resume_checkpoint"]))
            checkpoint = torch.load(cfg["resume_checkpoint"])
            cfg.opt["start_epoch"] = checkpoint["epoch"] - 1
            self.model.module.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint {} (epoch {})".format(cfg["resume"], checkpoint["epoch"]))
        
        self.best_loss = float("inf")

    def training(self, epoch):
        train_loss = 0.0
        train_mae = 0.0
        self.model.train()
        num_iter_train = len(self.train_dataloader)
        start_time = time.time()
        for i, data in enumerate(self.train_dataloader):
            images, labels = data[0].cuda(), data[1].unsqueeze(1).cuda()
            
            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()

            preds = self.model(images)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()
            mae = self.metric(preds, labels)
            train_loss += loss.item()
            train_mae += mae.item()
        end_time = time.time()

        train_loss = train_loss / num_iter_train
        train_mae = train_mae / num_iter_train
        print("Epoch: {:d}, Train loss: {:.5f}, Mean absolute error: {:.5f}".format(epoch, train_loss, train_mae))
        self.summary.add_scalar(
            "training loss", train_loss, epoch
        )
        self.summary.add_scalar(
            "training mae", train_mae, epoch
        )
        self.summary.add_scalar(
            "training time", end_time-start_time, epoch
        )
    
    def validation(self, epoch):
        test_loss = 0.0
        test_mae = 0.0
        self.model.eval()
        num_iter_val = len(self.val_dataloader)
        start_time = time.time()
        for i, data in enumerate(self.val_dataloader):
            images, labels = data[0].cuda(), data[1].unsqueeze(1).cuda()
            with torch.no_grad():
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                mae = self.metric(preds, labels)
                test_loss += loss.item()
                test_mae += mae.item()
            if i == 0:
                all_true_labels = labels.detach().cpu()
                all_predicted_labels = preds.detach().cpu()
            else:
                all_true_labels = torch.cat((all_true_labels, labels.detach().cpu()), dim=0)
                all_predicted_labels = torch.cat((all_predicted_labels, preds.detach().cpu()), dim=0)
        end_time = time.time()
        test_loss = test_loss / num_iter_val
        test_mae = test_mae / num_iter_val
        all_true_labels = (all_true_labels > self.cfg["ef_threshold"]).to(torch.long)
        auroc = roc_auc_score(all_true_labels, all_predicted_labels)
        auprc = average_precision_score(all_true_labels, all_predicted_labels)
        print("Epoch: {:d}, Test loss: {:.5f}, Mean absolute error: {:.5f}".format(epoch, test_loss, test_mae))
        print("AUROC: {:.5f}, AUPRC: {:.5f}".format(auroc, auprc))
        self.summary.add_scalar(
            "test loss", test_loss, epoch
        )
        self.summary.add_scalar(
            "test mae", test_mae, epoch
        )
        self.summary.add_scalar(
            "test auroc", auroc, epoch
        )
        self.summary.add_scalar(
            "test auprc", auprc, epoch
        )
        self.summary.add_scalar(
            "test time", end_time-start_time, epoch
        )
        
        is_best = False
        if test_loss < self.best_loss:
            is_best = True
            self.best_loss = test_loss
        self.saver.save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": test_loss,
            "best_mae": test_mae,
            "best_auroc": auroc,
            "best_auprc": auprc
        }, is_best, save_model=self.cfg["save_model"])

        # return self.early_stopper.early_stop(test_loss)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Fine-tuning for Echocardiograms"
    )
    parser.add_argument("--config", type=str, default=None, help="experiment configuration file")
    parser.add_argument("--saved_pretrained_model", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--num_modes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument("--combine", type=str, default=None, choices=["avg", "lstm", "concat"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_test", action="store_true")
    parser.add_argument("--percent", type=float, default=-1.0)
    parser.add_argument("--sample_mode_train", type=str, default="random", choices=["random", "fixed_random", "sequential"])
    parser.add_argument("--sample_mode_val", type=str, default="fixed_random", choices=["random", "fixed_random", "sequential"])
    parser.add_argument("--clf", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--num_clips", type=int, default=None)

    args = parser.parse_args()

    cfg = SupConfig(args.config)

    cfg.opt["lr"] = args.lr
    cfg.opt["epochs"] = args.epoch
    cfg.opt["train_batch_size"] = args.train_batch_size
    cfg.opt["saved_pretrained_model"] = args.saved_pretrained_model
    cfg.opt["freeze"] = args.freeze
    cfg.opt["normalize"] = args.normalize
    cfg.opt["num_modes"] = args.num_modes
    cfg.opt["pretrained"] = args.pretrained
    cfg.opt["in_channels"] = args.in_channels
    cfg.opt["encoder"] = args.encoder
    cfg.opt["combine"] = args.combine
    cfg.opt["seed"] = args.seed
    cfg.opt["use_test"] = args.use_test
    if args.use_test:
        cfg.opt["percent_val"] = args.percent
    else:
        cfg.opt["percent_train"] = args.percent
    cfg.opt["sample_mode_train"] = args.sample_mode_train
    cfg.opt["sample_mode_val"] = args.sample_mode_val
    cfg.opt["clf"] = args.clf
    cfg.opt["num_clips"] = args.num_clips
    cfg.set_environmental_variables()
    cfg.write_to_json()

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    trainer = Trainer(cfg)
    print("Starting Epoch:", trainer.cfg["start_epoch"])
    print("Total Epoches:", trainer.cfg["epochs"])
    for epoch in range(trainer.cfg["start_epoch"], trainer.cfg["epochs"]):
        isValidationEpoch = (epoch > cfg["eval_epoch_begin"] and (
            epoch + 1) % cfg["eval_interval"] == 0)
        trainer.training(epoch)
        if not trainer.cfg["no_val"] and isValidationEpoch == True:
            trainer.validation(epoch)
            # if early_stop:
            #     break
    
    trainer.summary.close()

if __name__ == "__main__":
    main()

        
