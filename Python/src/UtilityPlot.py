from matplotlib import pyplot as plt


def plot_time_series(x, y, label=None, xlabel='X', ylabel='Values', title=None, linestyle='-', color='blue'):
    plt.plot(x, y, label=label, linestyle=linestyle, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
