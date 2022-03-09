import sklearn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
    X, y: numpy array
    tr_val_te_ratio_array: [train, validation, test]
'''
def split_data(X, y, tr_val_te_ratio_array=[8, 1, 1]):
    if (len(tr_val_te_ratio_array) != 3):
        raise Exception("Ratio array should be [train, validation, test]")
    rarray = tr_val_te_ratio_array
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                test_size=rarray[2]/(rarray[0] + rarray[1] + rarray[2]),
                                                                random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=rarray[1]/(rarray[0] + rarray[1]),
                                                      random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

'''
    normalize_array: [fit_transform_element, ...]
'''
def normalize_data(normalize_array):
    if (normalize_array == None or len(normalize_array) == 0):
        raise Exception('normalize_array: [fit_transform_element, ...]')
    sc = StandardScaler()
    array_len = len(normalize_array)
    for i in range(array_len):
        if i == 0:
            normalize_array[i] = sc.fit_transform(normalize_array[i])
        else:
            normalize_array[i] = sc.transform(normalize_array[i])
    return normalize_array

def numpy_to_tensor(array):
    for i in range(len(array)):
        array[i] = torch.tensor(array[i].astype(np.float32))
    return array
