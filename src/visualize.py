import click

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import cuda
from torch.utils.data import DataLoader

import network
from dataset import WordsDataset
from utils import text


def graph_embeddings(embeddings, data):
    click.echo('Creating model')
    tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=1000, random_state=32, verbose=10)
    tsne_embeddings = np.array(tsne_model.fit_transform(embeddings))
    click.echo('Creating figure')
    plt.figure(figsize=(40, 40))
    x = tsne_embeddings[:, 0]
    y = tsne_embeddings[:, 1]
    plt.scatter(x, y)
    for i, word in enumerate(data.words):
        plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom', size=8)
    plt.grid(True)
    plt.show()


def difference_similarity(embeddings, data, pairs):
    es = []
    for singular, plural in pairs:
        i = data.words.index(singular)
        j = data.words.index(plural)
        se = embeddings[i, :].reshape(1, -1)
        pe = embeddings[j, :].reshape(1, -1)
        es.append((se, pe))

    for i, (sa, pa) in enumerate(es):
        delta = pa - sa
        print('{}-{},'.format(*pairs[i]), end='')
        for j, (sb, pb) in enumerate(es):
            s = cosine_similarity(sb + delta, pb)
            print('{},'.format(s.item()), end='')
        print('')


@click.command()
@click.option('--path', '-p', default='../dataset/corpus.txt', help='path of the dataset file')
@click.option('--limit', '-l', default=0, help='use only first l lines from the dataset file')
@click.option('--load-model', '-i', default=None, help='path to the model dict')
@click.option('--align', '-a', default=64, help='alignment of characters in a single word')
@click.option('--batch-size', '-b', default=100)
@click.option('--workers', '-w', default=2)
@click.option('--graph', is_flag=True, default=False)
@click.option('--check-relations', default=None)
def main(path: str, limit: int, load_model: str, align: int, batch_size: int, workers: int,
         graph: bool, check_relations: bool):
    # Check if cuda is available
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    # Load net
    encoder = text.Encoder(alignment=align)

    net = network.Language(alignment=align, alphabet_size=encoder.N, embedding_size=16)
    net.to(device)

    if load_model is not None:
        click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
        net.load_state_dict(torch.load(load_model, map_location=device))

    # Load dataset
    click.echo('Loading dataset \'{}\''.format(path))
    data = WordsDataset(path=path, encoder=encoder, limit=limit)
    loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=workers)

    # Create embeddings
    click.echo('Creating embbeddings')
    embeddings = torch.zeros(len(data), net.embedding_size)
    net.eval()
    with click.progressbar(loader, label='Embedding words') as bar:
        i = 0
        for inputs in bar:
            N = inputs.size(0)
            embeddings[i:i + N, :] = net.embedder(inputs)
            i += N

    embeddings = embeddings.detach().numpy()
    if check_relations is not None:
        with open(check_relations) as f:
            pairs = [tuple(line.strip().split(',')) for line in f]
            difference_similarity(embeddings, data, pairs)

    if graph:
        graph_embeddings(embeddings, data)


if __name__ == '__main__':
    main()
