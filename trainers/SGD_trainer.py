import torch
from loggers.trainer_loggers import cal_accuracy


'''
    train the machine learning model based on SGD
    return a tuple of 
        training loss, training accuracy, validation loss, validation accuracy 
    for each batch within the training process
     
    model: the machine learning model to be trained
    training array: (X_train, y_train, X_val, y_val) where
        X_train: training data
        y_train: training labels with ordinal encoding
        X_val: validation data
        y_val: validation labels with ordinal encoding
    device: 'gpu' if model in trained on GPU
            the trained model and data will be moved back to cpu after training
'''


def train(model, training_array, loss, optimizer, batch_size=100, num_epoch=40, device='cpu'):
    model.to(device)
    X_train, y_train, X_val, y_val = training_array
    X_train.to(device)
    y_train.to(device)
    X_val.to(device)
    y_val.to(device)
    training_loss_array = []
    validation_loss_array = []
    training_accuracy_array = []
    validation_accuracy_array = []
    for epoch in range(num_epoch):
        for i in range(0, len(X_train), batch_size):
            _input = X_train[i:i + batch_size]
            label = y_train[i:i + batch_size]
            pred = model(_input)
            l = loss(pred, label)
            training_loss_array.append(l.item())
            validation_loss_array.append(loss(model(X_val), y_val).item())
            training_accuracy_array.append(cal_accuracy(model, X_train, y_train))
            validation_accuracy_array.append(cal_accuracy(model, X_val, y_val))
            model.zero_grad()
            l.backward()
            optimizer.step()
        print("Epoch %2d: loss on final training batch: %.4f" % (epoch, l.item()))
        print("training accuracy: %.2f%% validation accuracy: %.2f%%" % (
            training_accuracy_array[-1] * 100, validation_accuracy_array[-1] * 100))
    model.to('cpu')
    X_train.to('cpu')
    X_val.to('cpu')
    y_train.to('cpu')
    y_val.to('cpu')
    return training_loss_array, training_accuracy_array, validation_loss_array, validation_accuracy_array
