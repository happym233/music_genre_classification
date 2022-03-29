import matplotlib.pyplot as plt

'''
    plot several numerical arrays within one figure (to show contract for each line)
    num_arrays: a set of numerical arrays [x1, x2, ..., xn]
    labels: labels for each line [l1, l2, ..., ln] (eg. l1 is the label of x1)
'''


def plot_numerical_arrays(num_arrays=[], labels=[], xlabel='', ylabel='', title=''):
    plt.figure(figsize=(20, 10))
    if len(num_arrays) != len(labels):
        raise Exception("length of numerical arrays should be same as length of labels")
    num_len = len(num_arrays)
    for i in range(num_len):
        plt.plot(num_arrays[i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


'''
    print statistics of input data
    modified from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
'''


def print_stats(data, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    print("Shape:", tuple(data.shape))
    print("Dtype:", data.dtype)
    print(f" - Max:     {data.max().item():6.3f}")
    print(f" - Min:     {data.min().item():6.3f}")
    print(f" - Mean:    {data.mean().item():6.3f}")
    print(f" - Std Dev: {data.std().item():6.3f}")
    print()
    print(data)
    print()
