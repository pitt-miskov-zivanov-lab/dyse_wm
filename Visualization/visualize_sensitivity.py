import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm


def plot_bar(data, data_label, figure_title, y_scale='linear', style='whitegrid'):
    sns.set_style(style)
    fig = plt.figure(figsize=(8,4))
    plt.bar(range(len(data_label)), data, align='center')
    plt.xticks(range(len(data_label)), data_label, rotation='vertical', fontsize='12')
    plt.title(figure_title)
    plt.ylabel('Value')
    plt.yscale(y_scale)
    fig.tight_layout()

    # close plot to suppress output, since figure is returned by this function
    plt.close()

    return fig


def plot_matrix(
        data,
        xticklabels,
        yticklabels,
        figure_title,
        color_scale='linear',
        cmap='Blues',
        style='white',
        font_size='12'):

    sns.set_style(style)

    fig, ax = plt.subplots(figsize=(10,8))
    if color_scale=='log':
        a = np.array(data)
        b = np.where(a>0,a,np.nan)
        im = ax.imshow(b, cmap=cmap, norm=LogNorm())
    else:
        im = ax.imshow(data, cmap=cmap, norm=None)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=font_size)
    ax.set_yticklabels(yticklabels, fontsize=font_size)
    ax.set_ylim([len(yticklabels)-0.5, -0.5])
    ax.set_xlabel('Target', fontsize=font_size)
    ax.set_ylabel('Source', fontsize=font_size)
    ax.set_title(figure_title)
    fig.tight_layout()

    # close plot to suppress output, since figure is returned by this function
    plt.close()

    return fig
