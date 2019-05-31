import click

import torch
from torch import cuda
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, random_split

from dataset import CorpusDataset
from preprocessor import CorpusPreprocessor
from utils import split_size, PadCollate, text


@click.command()
@click.option('--path', '-p', default='../dataset/corpus.txt', help='path of the dataset file')
@click.option('--limit', '-l', default=0, help='use only first l lines from the dataset file')
@click.option('--mask', '-m', default='?', help='text used to replace a masked word')
@click.option('--seed', '-s', default=None, help='seed for the PRNG')
@click.option('--align', default=48, help='alignment of characters in a single word')
@click.option('--validation', '-v', default=0.2, help='percentage of the dataset used for validation')
@click.option('--batch-size', '-b', default=1000)
@click.option('--workers', '-w', default=2, help='number of workers in the data loader')
def main(path: str, limit: int,
         mask: str, seed: int, align: int,
         validation: float, batch_size: int, workers: int) -> None:
    # Check if cuda is available
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    # Instantiate data sets
    click.echo('Loading dataset \'{}\', using {}% as validation dataset'.format(path, validation * 100))

    preprocessor = CorpusPreprocessor(mask=mask, seed=seed)
    encoder = text.Encoder(alignment=align)
    dataset = CorpusDataset(path=path, limit=limit, preprocessor=preprocessor, encoder=encoder)
    validation_data, train_data = random_split(dataset=dataset, lengths=split_size(validation, len(dataset)))

    # Instantiate data loaders
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=workers,
                              collate_fn=PadCollate(dim=0))
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, num_workers=workers,
                                   collate_fn=PadCollate(dim=0))

    for xs, ys in train_loader:
        print(xs.shape, ys.shape)


if __name__ == '__main__':
    main()
