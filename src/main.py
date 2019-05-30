import click

import torch
from torch import cuda
from torch.utils.data import DataLoader, random_split

from dataset import CorpusDataset
from preprocessor import CorpusPreprocessor
from utils import split_size, text


@click.command()
@click.option('--path', '-p', default='../dataset/corpus.txt', help='path of the dataset file')
@click.option('--limit', '-l', default=0, help='use only first l lines from the dataset file')
@click.option('--mask', '-m', default='?', help='text used to replace a masked word')
@click.option('--seed', '-s', default=None, help='seed for the PRNG')
@click.option('--encoding', '-e', default=48, help='initial encoding length')
@click.option('--validation', '-v', default=0.2, help='percentage of the dataset used for validation')
@click.option('--batch-size', '-b', default=1000)
@click.option('--workers', '-w', default=2, help='number of workers in the data loader')
def main(path: str, limit: int,
         mask: str, seed: int, encoding: int,
         validation: float, batch_size: int, workers: int) -> None:
    # Check if cuda is available
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    # Instantiate data sets
    click.echo('Loading dataset \'{}\', using {}% as validation dataset'.format(path, validation * 100))

    preprocessor = CorpusPreprocessor(mask=mask, seed=seed)
    encoder = text.TensorEncoder(encoding_length=encoding)

    dataset = CorpusDataset(path=path, limit=limit, preprocessor=preprocessor, encoder=encoder)
    validation_data, train_data = random_split(dataset=dataset, lengths=split_size(validation, len(dataset)))

    # Instantiate data loaders
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=workers)
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, num_workers=workers)

    for word in train_loader:
        print(word)


if __name__ == '__main__':
    main()
