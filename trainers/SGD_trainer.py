import torch
from loggers.trainer_loggers import cal_accuracy

'''
    training array: [X_train, y_train, X_val, y_val]
'''


def train(model, training_array, batch_size=100, num_epoch=40, lr=0.01):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    X_train, y_train, X_val, y_val = training_array
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
    return training_loss_array, training_accuracy_array, validation_loss_array, validation_accuracy_array
