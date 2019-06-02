import os
import time

import click

import torch
from torch import cuda
from torch.utils.data import DataLoader, random_split

from dataset import CorpusDataset
from preprocessor import CorpusPreprocessor
from utils import split_size, PadCollate, text
from classifier import Classifier
import network


@click.command()
@click.option('--path', '-p', default='../dataset/corpus.txt', help='path of the dataset file')
@click.option('--limit', '-l', default=0, help='use only first l lines from the dataset file')
@click.option('--load-model', '-i', default=None, help='path to the model dict')
@click.option('--save-model', '-o', default='../output/', help='path to the model dict')
@click.option('--learning-rate', '-r', default=0.1, help='learning rate of the model')
@click.option('--epochs', '-e', default=5, help='number of epochs')
@click.option('--mask', '-m', default='?', help='text used to replace a masked word')
@click.option('--seed', '-s', default=None, help='seed for the PRNG')
@click.option('--align', '-a', default=64, help='alignment of characters in a single word')
@click.option('--validation', '-v', default=0.2, help='percentage of the dataset used for validation')
@click.option('--batch-size', '-b', default=100)
@click.option('--workers', '-w', default=2, help='number of workers in the data loader')
@click.option('--no-train', is_flag=True, default=False)
def main(path: str, limit: int,
         load_model: str, save_model: str, learning_rate: float, epochs: int,
         mask: str, seed: int, align: int,
         validation: float, batch_size: int, workers: int,
         no_train: bool) -> None:
    # Check if cuda is available
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    preprocessor = CorpusPreprocessor(mask=mask, seed=seed)
    encoder = text.Encoder(alignment=align)

    # Load net
    net = network.Language(alignment=align, alphabet_size=encoder.N, embedding_size=16)
    net.to(device)

    if load_model is not None:
        click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
        net.load_state_dict(torch.load(load_model, map_location=device))

    # Instantiate data sets
    click.echo('Loading dataset \'{}\', using {}% as validation dataset'.format(path, validation * 100))

    dataset = CorpusDataset(path=path, limit=limit, preprocessor=preprocessor, encoder=encoder)
    validation_data, train_data = random_split(dataset=dataset, lengths=split_size(validation, len(dataset)))

    # Instantiate data loaders
    pad_collate = PadCollate(dim=0)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=pad_collate)
    test_loader = DataLoader(
        dataset=validation_data, batch_size=batch_size, num_workers=workers, collate_fn=pad_collate)

    # Run training
    classifier = Classifier(net, lr=learning_rate)

    if not no_train:
        click.secho('Training model', fg='blue')
        net.train()
        classifier.train(train_loader, test_loader, epochs)
    else:
        click.secho('Testing model', fg='blue')
        net.eval()
        classifier.test(test_loader)

    if save_model is not None and not no_train:
        suffix = time.strftime("%Y%m%d-%H%M%S")
        name = 'model_' + suffix
        click.secho('Saving model as \'{}\''.format(name), fg='yellow')
        torch.save(net.state_dict(), os.path.join(save_model, name))


if __name__ == '__main__':
    main()
