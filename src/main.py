import click

import torch
from torch import cuda
from torch.utils.data import DataLoader, random_split

from dataset.text import TextDataset


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
    click.echo('Loading dataset from \'{}\', using {}% as validation dataset'.format(dataset, validation * 100))

    data = TextDataset(path=dataset, mask=mask, limit=limit, seed=seed)
    N = int(validation * len(data))
    test_data, train_data = random_split(dataset=data, lengths=[N, len(data) - N])

    # Instantiate data loaders
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=workers)
    test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size, num_workers=workers)


if __name__ == '__main__':
    main()
