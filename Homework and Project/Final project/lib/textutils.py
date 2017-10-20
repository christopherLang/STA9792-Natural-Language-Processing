import random


def data_splitter(length, prop, seed=12345):
    random.seed(seed)

    if isinstance(length, int) is not True:
        raise TypeError("parameter length should be an integer")

    sizes = [round(i * length) for i in prop]

    if sum(sizes) != length:
        while True:
            if sum(sizes) == length:
                break
            else:
                index = random.sample(range(len(sizes)), 1)[0]
                sizes[index] -= 1

    document_indices = list(range(length))
    random.shuffle(document_indices)

    result = list()
    initial_i = 0
    for data_size in sizes:
        r = [document_indices[i] for i in range(initial_i,
                                                initial_i + data_size)]
        result.append(r)

        initial_i = data_size

    return result
