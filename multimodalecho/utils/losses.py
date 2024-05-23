from math import ceil, floor
import torch
import torch.nn as nn


# Loss used for self-paced learning (regression)
class SPLLoss_regression(nn.MSELoss):
    def __init__(self, *args, threshold=None, growing_factor=None, quantile=None, **kwargs):
        super(SPLLoss_regression, self).__init__(*args, **kwargs)
        # initial loss threshold
        self.threshold = threshold
        # factor by which threshold is increased each epoch
        self.growing_factor = growing_factor
        self.quantile = quantile
        self.train_size = 0

    def forward(self, input, target):
        # compute loss for each sample -> no reduction
        super_loss = nn.functional.mse_loss(input, target, reduction="none")

        # use quantiles to find reasonable threshold
        if self.quantile is not None:
            self.threshold = torch.quantile(super_loss, self.quantile)
        
        v = self.spl_loss(super_loss)
        self.train_size += torch.sum(v)
        # only samples with loss < threshold contribute to total loss
        return (super_loss * v).mean()

    def increase_threshold(self):
        if self.threshold is not None:
            self.threshold *= self.growing_factor

    def get_threshold(self):
        return self.threshold

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.to(torch.int8)

    def get_train_size(self):
        train_size = self.train_size
        self.train_size = 0 # reset after each epoch
        return train_size


# Loss used for vanilla curriculum learning
class CLLoss_regression(nn.MSELoss):
    def __init__(self, *args, batch_size=0, num_epochs=0, init_fraction=0, epoch_fraction=0, 
        pacing_func="linear", **kwargs):
        super(CLLoss_regression, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pacing_func = pacing_func
        # Compute initial train set size and earliest epoch where full train set is used
        self.init_size = ceil(batch_size * init_fraction)
        self.epoch_full = ceil(num_epochs * epoch_fraction)
        self.train_size = 0

    def forward(self, epoch, input, target):
        # compute loss for each sample -> no reduction
        super_loss = nn.functional.mse_loss(input, target, reduction="none")
        # select desired fraction of samples, depending on their loss
        if self.pacing_func == "linear":
            size = floor(self.init_size + ((self.batch_size - self.init_size) / self.epoch_full * epoch))
            size = min(size, self.batch_size)
            loss = torch.topk(super_loss, size, largest=False).values
            self.train_size += size
        elif self.pacing_func == "step":
            if epoch >= self.epoch_full: # use entire train set
                loss = super_loss
                self.train_size += self.batch_size
            else: # use initial train set size
                loss = torch.topk(super_loss, self.init_size, largest=False).values
                self.train_size += self.init_size
        
        return loss.mean()

    def get_train_size(self):
        train_size = self.train_size
        self.train_size = 0 # reset after each epoch
        return train_size
    