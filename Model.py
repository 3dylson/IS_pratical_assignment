import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
from MArrhythmiaRNN import MArrhythmiaRNN
import pandas as pd
from sklearn.linear_model import LinearRegression

finalFile =None

def dataPrep()
    for i in range(100, 234):
        try:
            file = pd.read_csv("datasets/" + str(i) + ".csv")

            finalFile = pd.concat([finalFile, file])
        except FileNotFoundError:
            continue
    return finalFile

def trainTest(file)
    train_df = train_test_split(
      file,
      test_size=0.33,
      random_state=RANDOM_SEED
    )
    test_df = train_test_split(
      val_df,
      test_size=06,
      random_state=RANDOM_SEED
    )
    return train_df, test_df


# read the dataset
    df = file
    # get the locations
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=0)

    train = pd.DataFrame(iris.data)
    test = pd.DataFrame(iris.target)


class Model(nn.Module):

    n_epochs = 100
    n_iters = 50
    hidden_size = 10

    model = MArrhythmiaRNN(hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = np.zeros(n_epochs) # For plotting

    for epoch in range(n_epochs):

        for iter in range(n_iters):
            _inputs = sample(50)
            inputs = Variable(torch.from_numpy(_inputs[:-1]).float())
            targets = Variable(torch.from_numpy(_inputs[1:]).float())

            # Use teacher forcing 50% of the time
            force = random.random() < 0.5
            outputs, hidden = model(inputs, None, force)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses[epoch] += loss.data[0]

        if epoch > 0:
            print(epoch, loss.data[0])

        # Use some plotting library
        # if epoch % 10 == 0:
            # show_plot('inputs', _inputs, True)
            # show_plot('outputs', outputs.data.view(-1), True)
            # show_plot('losses', losses[:epoch] / n_iters)

            # Generate a test
            # outputs, hidden = model(inputs, False, 50)
            # show_plot('generated', outputs.data.view(-1), True)

    # Online training
    hidden = None

    while True:
        inputs = get_latest_sample()
        outputs, hidden = model(inputs, hidden)

        optimizer.zero_grad()
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
