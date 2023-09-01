import torch
import numpy as np
import os
import argparse
import random
import time

from models.clnet import CLNet
from dataloaders import make_data_loader
from utils.loss import PAConLoss, SAConLoss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from config.ssl_config import SSLConfig

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.saver = Saver(cfg)
        self.summary = TensorboardSummary(cfg["log_dir"])

        kwargs = {"num_workers": cfg["num_workers"], "pin_memory": True}
        self.train_dataloader, self.val_dataloader = make_data_loader(cfg, **kwargs)

        model = CLNet(encoder=cfg["encoder"],
                      pretrained=cfg["pretrained"], 
                      linear_layers=cfg["linear_layers"],
                      in_channels=cfg["in_channels"],
                      normalize=cfg["normalize"]
                      )
        
        if cfg["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                        weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
        elif cfg["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                         weight_decay=cfg["weight_decay"], amsgrad=True)
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        self.model = model
        self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

        self.temperature = cfg["temperature"]

        self.criterion1 = PAConLoss(temperature=self.temperature)
        self.criterion2 = SAConLoss(temperature=self.temperature)
        self.gamma = cfg["gamma"] # weight of classification loss

        self.scale_by_temperature = cfg["scale_by_temperature"]
        self.epochs = cfg["epochs"]

        self.scheduler = LR_Scheduler(mode=cfg["lr_scheduler"], base_lr=cfg["lr"],
                                num_epochs=cfg["epochs"], iters_per_epoch=len(
                                    self.train_dataloader),
                                lr_step=cfg["lr_step"], warmup_epochs=cfg["warmup_epoch"])

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
        train_loss1 = 0.0
        train_loss2 = 0.0
        self.model.train()
        num_iter_train = len(self.train_dataloader)

        start_time = time.time()
        for i, data in enumerate(self.train_dataloader):
            images = data[0].cuda()
            if self.gamma > 0:
                aug_images = data[1].cuda()

            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()

            features = self.model(images)
            assert features.size(1) == 1
            features = features.squeeze(dim=1)
            if self.gamma > 0:
                aug_features = self.model(aug_images) # (batch_size, num_clips, num_modes, feature_size)
                assert aug_features.size(1) == 1
                aug_features = aug_features.squeeze(dim=1)  # (batch_size, num_modes, feature_size)
                if self.gamma < 1:
                    comb_features = torch.stack((features, aug_features), dim=2) # (batch_size, num_modes, 2, feature_size)
                else:
                    comb_features = torch.stack((features.reshape((-1, features.shape[-1])), aug_features.reshape((-1, aug_features.shape[-1]))), dim=1) # (batch_size*num_modes, 2, feature_size)

            if self.gamma == 1:
                loss = self.criterion1(comb_features)
            elif self.gamma == 0:
                loss = self.criterion1(features)
            else:
                loss1 = self.criterion1(features)
                loss2 = self.criterion2(comb_features)
                loss = (1 - self.gamma) * loss1 + self.gamma * loss2

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if self.gamma > 0 and self.gamma < 1:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()

        end_time = time.time()
        train_loss = train_loss / num_iter_train
        if self.gamma > 0 and self.gamma < 1:
            train_loss1 = train_loss1 / num_iter_train
            train_loss2 = train_loss2 / num_iter_train
        print("Epoch: {:d}, Train loss: {:.5f}".format(epoch, train_loss))
        if self.gamma > 0 and self.gamma < 1:
            print("train LPA: {:.5f}".format(train_loss1))
            print("train LSA: {:.5f}".format(train_loss2))

        self.summary.add_scalar("training loss", train_loss, epoch)
        self.summary.add_scalar("training time", end_time-start_time, epoch)
        if self.gamma > 0 and self.gamma < 1:
            self.summary.add_scalar("train LPA", train_loss1, epoch)
            self.summary.add_scalar("train LSA", train_loss2, epoch)
    
    def validation(self, epoch):
        test_loss = 0.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        self.model.eval()
        num_iter_val = len(self.val_dataloader)
        start_time = time.time()
        for i, data in enumerate(self.val_dataloader):
            images = data[0].cuda()
            if self.gamma > 0:
                aug_images = data[1].cuda()

            with torch.no_grad():
                features = self.model(images)
                assert features.size(1) == 1
                features = features.squeeze(dim=1)
                if self.gamma > 0:
                    aug_features = self.model(aug_images)
                    assert aug_features.size(1) == 1
                    aug_features = aug_features.squeeze(dim=1)
                    if self.gamma < 1:
                        comb_features = torch.stack((features, aug_features), dim=2)
                    else:
                        comb_features = torch.stack((features.reshape((-1, features.shape[-1])), aug_features.reshape((-1, aug_features.shape[-1]))), dim=1)

                if self.gamma == 1:
                    loss = self.criterion1(comb_features)
                elif self.gamma == 0:
                    loss = self.criterion1(features)
                else:
                    loss1 = self.criterion1(features)
                    loss2 = self.criterion2(comb_features)
                    loss = (1 - self.gamma) * loss1 + self.gamma * loss2

                test_loss += loss.item()
                if self.gamma > 0 and self.gamma < 1:
                    test_loss1 += loss1.item()
                    test_loss2 += loss2.item()
        end_time = time.time()
        test_loss = test_loss / num_iter_val
        if self.gamma > 0 and self.gamma < 1:
            test_loss1 = test_loss1 / num_iter_val
            test_loss2 = test_loss2 / num_iter_val
        print("Epoch: {:d}, Test loss: {:.5f}".format(epoch, test_loss))
        if self.gamma > 0 and self.gamma < 1:
            print("test LPA: {:.5f}".format(test_loss1))
            print("test LSA: {:.5f}".format(test_loss2))
        self.summary.add_scalar("test loss", test_loss, epoch)
        self.summary.add_scalar("test time", end_time-start_time, epoch)
        if self.gamma > 0 and self.gamma < 1:
            self.summary.add_scalar("test LPA", test_loss1, epoch)
            self.summary.add_scalar("test LSA", test_loss2, epoch)
        
        is_best = False
        if test_loss < self.best_loss:
            is_best = True
            self.best_loss = test_loss
        self.saver.save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": test_loss,
            "best_lna": test_loss1,
            "best_lnd": test_loss2
        }, is_best, save_model=self.cfg["save_model"])

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Contrastive Learning for Echocardiograms"
    )
    parser.add_argument("--config", type=str, default=None, help="experiment configuration file")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--num_modes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument("--combine", type=str, default=None, choices=["mean", "lstm", "concat"])
    parser.add_argument("--warmup_epoch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--sample_mode_train", type=str, default="random", choices=["random", "fixed_random", "sequential"])
    parser.add_argument("--sample_mode_val", type=str, default="fixed_random", choices=["random", "fixed_random", "sequential"])
    parser.add_argument("--nb_thresh", type=int, default=50)
    parser.add_argument("--aug", type=str, default="mix")
    parser.add_argument("--num_clips", type=int, default=None)

    args = parser.parse_args()

    cfg = SSLConfig(args.config)
    if args.pretrained:
        cfg.opt["pretrained"] = True
    else:
        cfg.opt["pretrained"] = False
    cfg.opt["lr"] = args.lr
    cfg.opt["epochs"] = args.epoch
    cfg.opt["train_batch_size"] = args.train_batch_size
    cfg.opt["num_modes"] = args.num_modes
    cfg.opt["in_channels"] = args.in_channels
    cfg.opt["temperature"] = args.temperature
    cfg.opt["encoder"] = args.encoder
    cfg.opt["combine"] = args.combine
    cfg.opt["warmup_epoch"] = args.warmup_epoch
    cfg.opt["seed"] = args.seed
    cfg.opt["gamma"] = args.gamma
    cfg.opt["sample_mode_train"] = args.sample_mode_train
    cfg.opt["sample_mode_val"] = args.sample_mode_val
    cfg.opt["nb_thresh"] = args.nb_thresh
    cfg.opt["aug"] = args.aug
    cfg.opt["normalize"] = not args.no_normalize
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
    
    trainer.summary.close()

if __name__ == "__main__":
    main()

        
