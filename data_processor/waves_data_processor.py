import torchaudio
import torch

'''
    return the processed (data, label) with
        music_filename_lists: a list of music file name that requires to be processed
        mini_frag_sec: with each origin fragments are too long(30s), one fragments is split
            into minor parts with shorter length(mini_frag_sec)
        feature_extraction_func: a function which take the waveform as input and output the processed features
'''


def generate_wave_features(music_filename_lists, frag_len, feature_extraction_fun,
                           root='original_data/genres_original/'):
    genre_num = len(music_filename_lists)
    features = None
    labels = None
    for label in range(genre_num):
        for music in music_filename_lists[label]:
            waveform, sample_rate = torchaudio.load(root + music)
            frag_num = int(waveform.shape[1] / (frag_len * sample_rate))
            median = int(waveform.shape[1] / 2)
            # split wave into two minor fragments
            frag_list = [waveform[0:1, i * frag_len * sample_rate: (i + 1) * frag_len * sample_rate]
                         for i in range(0, frag_num)]
            # print(frag_list[0].shape)
            wave_fragment_features = [feature_extraction_fun(frag) for frag in frag_list]
            if features is None:
                features = torch.cat(wave_fragment_features, dim=0)
            else:
                features = torch.cat((features, torch.cat(wave_fragment_features, dim=0)), dim=0)
            if labels is None:
                labels = torch.Tensor([label for i in range(frag_num)])
            else:
                labels = torch.cat((labels, torch.Tensor([label for i in range(frag_num)])), dim=0)
    return features, labels
