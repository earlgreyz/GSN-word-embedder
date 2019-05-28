import click


@click.command()
@click.argument('--dataset', '-d', default='../dataset/corpus.txt')
def main(dataset: str) -> None:
    pass


if __name__ == '__main__':
    main()
