from typing import List

import click

import torch
from torch import cuda
from torch.optim import adam
import torch.nn.functional as F

from callbacks import Callback
from utils import RunningAverage


class Classifier:
    def __init__(self, net, lr: float = .1, desired_accuracy: float = .6, callbacks: List[Callback] = None):
        self.net = net
        self.optimizer = adam.Adam(net.parameters(), lr=lr)
        self.desired_accuracy = desired_accuracy
        self.criterion = F.cross_entropy
        self.callbacks = []
        if callbacks is not None:
            self.callbacks = callbacks

    def train(self, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            click.echo('Training epoch {}'.format(epoch))
            self.net.train()
            loss = self._train_epoch(epoch=epoch, loader=train_loader)
            click.echo('Testing epoch {}'.format(epoch))
            self.net.eval()
            accuracy = self.test(test_loader)
            for callback in self.callbacks:
                callback(net=self.net, epoch=epoch, loss=loss, accuracy=accuracy)

    def _train_epoch(self, epoch, loader) -> float:
        running_loss = RunningAverage()
        show_stats = lambda _: '[{}, {:3f}]'.format(epoch + 1, running_loss.average)

        with click.progressbar(loader, item_show_func=show_stats) as bar:
            for inputs, lengths, targets in bar:
                if cuda.is_available():
                    inputs, lengths, targets = inputs.to('cuda'), lengths.to('cuda'), targets.to('cuda')

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs, lengths)
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
            for inputs, lengths, targets in bar:
                if cuda.is_available():
                    inputs, lengths, targets = inputs.to('cuda'), lengths.to('cuda'), targets.to('cuda')

                outputs = self.net(inputs, lengths)
                _, predicted = torch.max(outputs.data, 1)

                correct = (predicted == targets)
                accuracy.update(correct.sum().item(), targets.size(0))

        color = 'green' if accuracy.average > self.desired_accuracy else 'red'
        click.secho('Accuracy={}'.format(accuracy.average), fg=color)
        return accuracy.average
