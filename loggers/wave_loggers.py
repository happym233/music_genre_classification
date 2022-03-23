import matplotlib
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"""
    plot a wave based on waveform and sample_rate produced by torchvision after reading a .wav file
    modified from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
"""


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


'''
    a function that help plotting more visible for spectrogram
'''


def power_to_db(spec):
    return 10 * torch.log10(spec)

'''
    plot a speechbrain extracted spectrogram
'''


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    axs.yaxis.set_major_locator(MultipleLocator(int(spec.shape[0] / 200) * 40))
    axs.xaxis.set_major_locator(MultipleLocator(int(spec.shape[1] / 300) * 20))
    im = axs.imshow(power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


'''
    plot a speechbrain extracted filter bank
'''

def plot_filterbank(fbank, title=None, ylabel='mel_filter_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))
    axs.set_title(title or 'filter bank')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    axs.yaxis.set_major_locator(MultipleLocator(4))
    axs.xaxis.set_major_locator(MultipleLocator(int(fbank.shape[1] / 300) * 20))
    im = axs.imshow(fbank, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
