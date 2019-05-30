def split_size(percentage, size):
    N = int(percentage * size)
    M = size - N
    return N, M