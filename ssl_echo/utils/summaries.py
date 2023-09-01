import os 
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir = os.path.join(self.directory))
    def add_scalar(self, name, data, step):
        self.writer.add_scalar(name, data, step)
    def close(self):
        self.writer.close()