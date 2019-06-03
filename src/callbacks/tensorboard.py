from torch.utils.tensorboard import SummaryWriter

from .callback import Callback


class LoggerCallback(Callback):
    def __init__(self, logs_path):
        self.logs_path = logs_path

    def __call__(self, epoch, loss, accuracy, *args, **kwargs):
        with SummaryWriter(self.logs_path) as writer:
            writer.add_scalar('data/loss', loss, epoch)
            writer.add_scalar('data/accuracy', accuracy, epoch)
