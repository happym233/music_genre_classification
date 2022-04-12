def test_wave_data_loading():
    from data_processor.utils import load_numpy_arrays
    X_train, y_train, X_val, y_val, X_test, y_test = load_numpy_arrays(
        ['data_train_X.npy', 'data_train_y.npy', 'data_val_X.npy', 'data_val_y.npy', 'data_test_X.npy',
         'data_test_y.npy']
        , path_prefix='../processed_data/processed_waves/10-genres/')

    assert X_train.shape[1] == 501 and X_train.shape[2] == 40
    assert X_val.shape[1] == 501 and X_val.shape[2] == 40
    assert X_test.shape[1] == 501 and X_test.shape[2] == 40
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_csv_data_loading():
    from data_processor.utils import load_numpy_arrays
    data_train, data_val, data_test = load_numpy_arrays(['data_train.npy', 'data_val.npy', 'data_test.npy'],
                                                        path_prefix='processed_data/processed_features/10-genres/')
    assert data_train.shape[1] == 58
    assert data_val.shape[1] == 58
    assert data_test.shape[1] == 58
