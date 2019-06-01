import click
import torch
from torch import cuda

from torch.optim import adam
import torch.nn.functional as F

from utils import RunningAverage


class Classifier:
    def __init__(self, net, lr=0.1):
        self.net = net
        self.optimizer = adam.Adam(net.parameters(), lr=lr)
        self.criterion = F.cross_entropy

    def train(self, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            click.echo('Training epoch {}'.format(epoch))
            self.net.train()
            self._train_epoch(epoch=epoch, loader=train_loader)
            click.echo('Testing epoch {}'.format(epoch))
            self.net.eval()
            self.test(test_loader)

    def _train_epoch(self, epoch, loader) -> float:
        running_loss = RunningAverage()
        show_stats = lambda _: '[{}, {:3f}]'.format(epoch + 1, running_loss.average)

        with click.progressbar(loader, item_show_func=show_stats) as bar:
            for inputs, targets in bar:
                if cuda.is_available():
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs, _ = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss.update(loss.item())

        return running_loss.average

    def test(self, loader) -> float:
        accuracy = RunningAverage()
        show_stats = lambda _: '[{:2f}]'.format(accuracy.average)

        with click.progressbar(loader, item_show_func=show_stats) as bar:
            for inputs, targets in bar:
                if cuda.is_available():
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')

                outputs, _ = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                correct = (predicted == targets)
                accuracy.update(correct.sum().item(), targets.size(0))

        return accuracy.average
