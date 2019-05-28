import click

from torch.utils.data import DataLoader, random_split

from dataset.text import TextDataset


@click.command()
@click.option('--dataset', '-d', default='../dataset/corpus.txt', help='path of the dataset file')
@click.option('--mask', '-m', default='?', help='text used to replace a masked word')
@click.option('--limit', '-l', default=0, help='use only first L lines of the whole dataset')
@click.option('--seed', '-s', default=None, help='seed for the PRNG')
@click.option('--validation', '-v', default=0.2, help='percentage of the dataset used for validation')
@click.option('--workers', '-w', default=2, help='number of workers in the data loader')
def main(dataset: str, mask: str, limit: int, seed: int,
         validation: float, workers: int) -> None:
    # Instantiate datasets
    data = TextDataset(path=dataset, mask=mask, limit=limit, seed=seed)
    N = int(validation * len(data))
    train_data, test_data = random_split(dataset=data, lengths=(len(data) - N, N))

    # Instantiate dataloaders
    train_loader = DataLoader(dataset=train_data, num_workers=workers)
    test_loader = DataLoader(dataset=test_data, num_workers=workers)

    for x in train_data:
        print(x)

if __name__ == '__main__':
    main()
