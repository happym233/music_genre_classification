import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import OneHotEncoder

def load_wave_csv(file_path, labels=['blues', 'classical', 'country', 'disco']):
    wave_df = pd.read_csv(file_path)
    wave_df = wave_df.drop(labels="filename", axis=1)
    wave_df = wave_df[wave_df['label'].isin(labels)]
    enc = OneHotEncoder(handle_unknown='ignore')
    y = enc.fit_transform(wave_df[['label']]).toarray()
    X = wave_df.drop(['label', 'length'], axis=1).to_numpy()
    return X, y