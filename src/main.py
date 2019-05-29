import click

import torch
from torch import cuda
from torch.utils.data import DataLoader, random_split

from dataset.corpus import Corpus
from dataset.words import WordsDataset


def split_size(percentage, size):
    N = int(percentage * size)
    M = size - N
    return N, M


@click.command()
@click.option('--dataset', '-d', default='../dataset/corpus.txt', help='path of the dataset file')
@click.option('--mask', '-m', default='?', help='text used to replace a masked word')
@click.option('--limit', '-l', default=0, help='use only first l lines from the dataset file')
@click.option('--seed', '-s', default=None, help='seed for the PRNG')
@click.option('--validation', '-v', default=0.2, help='percentage of the dataset used for validation')
@click.option('--batch-size', '-b', default=1000)
@click.option('--workers', '-w', default=2, help='number of workers in the data loader')
def main(dataset: str, mask: str, limit: int, seed: int,
         validation: float, batch_size: int, workers: int) -> None:
    # Check if cuda is available
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    # Instantiate data sets
    click.echo('Loading dataset \'{}\', using {}% as validation dataset'.format(dataset, validation * 100))

    corpus = Corpus(path=dataset, mask=mask, seed=seed, limit=limit)
    words_dataset = WordsDataset(corpus=corpus)
    words_test, words_train = random_split(dataset=words_dataset, lengths=split_size(validation, len(words_dataset)))

    # Instantiate data loaders
    words_train_loader = DataLoader(dataset=words_train, shuffle=True, batch_size=batch_size, num_workers=workers)
    words_test_loader = DataLoader(dataset=words_test, shuffle=True, batch_size=batch_size, num_workers=workers)

    for word in words_train_loader:
        print(word)


if __name__ == '__main__':
    main()
