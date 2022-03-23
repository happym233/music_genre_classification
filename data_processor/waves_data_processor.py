import torchaudio
import torch


'''
    return the processed (data, label) with
        music_filename_lists: a list of music file name that requires to be processed
        mini_frag_sec: with each origin fragments are too long(30s), one fragments is split
            into two minor parts with shorter length(mini_frag_sec)
            e.g. mini_frag_sec = 10, the function will process (5s -> 15s) (15s -> 30s) of the original fragment
        feature_extraction_func: a function which take the waveform as input and output the processed features
'''

def generate_wave_features(music_filename_lists,  mini_frag_sec, feature_extraction_fun,
                           root='original_data/genres_original/'):
    genre_num = len(music_filename_lists)
    features = None
    labels = None
    for label in range(genre_num):
        for music in music_filename_lists[label]:
            waveform, sample_rate = torchaudio.load(root + music)
            median = int(waveform.shape[1] / 2)
            # split wave into two minor fragments
            wave_fragment_1 = waveform[0:, median - mini_frag_sec * sample_rate: median]
            wave_fragment_2 = waveform[0:, median: median + mini_frag_sec * sample_rate]
            wave_fragment_1_features = feature_extraction_fun(wave_fragment_1)
            wave_fragment_2_features = feature_extraction_fun(wave_fragment_2)
            if features is None:
                features = torch.cat((wave_fragment_1_features, wave_fragment_2_features), dim=0)
            else:
                features = torch.cat((features, wave_fragment_1_features, wave_fragment_2_features), dim=0)
            if labels is None:
                labels = torch.Tensor([label, label])
            else:
                labels = torch.cat((labels, torch.Tensor([label, label])), dim=0)
    return features, labels
