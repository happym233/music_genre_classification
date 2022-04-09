import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


'''
    load features with file(features_3_sec.csv or features_30_sec.csv)
    and output the processed data X and ordinal encoding label y
    
    file_path: str
        The file path of feature_x_seconds.csv
    labels: list 
        labels of music fragments that need to be selected(for 4-genres and 10-genres)
'''


def load_wave_csv(file_path, labels=['blues', 'classical', 'country', 'disco']):
    wave_df = pd.read_csv(file_path)
    wave_df = wave_df.drop(labels="filename", axis=1)
    wave_df = wave_df[wave_df['label'].isin(labels)]
    enc = OrdinalEncoder(handle_unknown='error')
    y = enc.fit_transform(wave_df[['label']])
    X = wave_df.drop(['label', 'length'], axis=1).to_numpy()
    return X, y
