import sklearn
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
    a function that take data as input and return training, validation and testing data
    X: numpy.array
        data
    y: numpy.array
        label
    tr_val_te_ratio_array: train, validation, test ratio
        train: validation: test = array[0]: array[1]:array[2]
'''


def split_data(X, y, tr_val_te_ratio_array=[8, 1, 1]):
    if len(tr_val_te_ratio_array) != 3:
        raise Exception("Ratio array should be [train, validation, test]")
    rarray = tr_val_te_ratio_array
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                test_size=rarray[2] / (
                                                                        rarray[0] + rarray[1] + rarray[2]),
                                                                random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=rarray[1] / (rarray[0] + rarray[1]),
                                                      random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


'''
    a normalization function which fit the normalizer based on the first element of normalize_array
        and transform all elements
    normalize_array: list
        [fit_transform_element, ...]
'''


def normalize_data(normalize_array):
    if normalize_array is None or len(normalize_array) == 0:
        raise Exception('normalize_array: [fit_transform_element, ...]')
    sc = StandardScaler()
    array_len = len(normalize_array)
    fit_ele = normalize_array[0]
    # normalize 2D array
    if len(fit_ele.shape) == 2:
        for i in range(array_len):
            if i == 0:
                normalize_array[i] = sc.fit_transform(normalize_array[i])
            else:
                normalize_array[i] = sc.transform(normalize_array[i])

    # normalize 3D array
    else:
        scalers = {}
        for i in range(array_len):
            normalized_ele = normalize_array[i]
            if i == 0:
                # create scaler for each channel
                for j in range(fit_ele.shape[1]):
                    scalers[j] = StandardScaler()
                    normalized_ele[:, j, :] = scalers[j].fit_transform(normalized_ele[:, j, :])
            else:
                for j in range(fit_ele.shape[1]):
                    normalized_ele[:, j, :] = scalers[j].fit_transform(normalized_ele[:, j, :])
            normalize_array[i] = normalized_ele
    if len(normalize_array) == 1:
        return normalize_array[0]
    return normalize_array


'''
    take an array of numpy of input and transform them into pytorch.Tensor with type float32
'''


def numpy_to_tensor(array):
    for i in range(len(array)):
        array[i] = torch.tensor(array[i].astype(np.float32))
    if len(array) == 1:
        return array[0]
    return array


'''
    save an numpy array on the designated path with type .npy
'''


def save_numpy_arrays(arrays, paths, path_prefix=''):
    if arrays is None or paths is None or len(arrays) != len(paths):
        raise Exception('array length should be same size of path length')
    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)
    l = len(arrays)
    paths = [path_prefix + path for path in paths]
    for i in range(0, l):
        np.save(paths[i], arrays[i])


'''
    load an numpy array on the designated path with type .npy
'''


def load_numpy_arrays(paths, path_prefix=''):
    if paths is None:
        return None
    paths = [path_prefix + path for path in paths]
    res = []
    for path in paths:
        res.append(np.load(path))
    if len(res) == 1:
        return res[0]
    return res


'''
    combine an array of functions and output one function 
        (for signal processing STFT -> cal_spectrogram -> cal_filterbank)
    func_array: an array of function [f1, f2, ..., fn]
    output: fn( ...f2(f1(x)))
'''


class FunctionArrayExecutor:

    def __init__(
            self,
            func_array=None
    ):
        self.func_array = func_array

    def __call__(self, X):
        if self.func_array is None or len(self.func_array) == 0:
            return X
        else:
            res = X
            for func in self.func_array:
                res = func(res)
            return res
