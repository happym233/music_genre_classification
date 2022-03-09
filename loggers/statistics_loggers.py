import matplotlib.pyplot as plt

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