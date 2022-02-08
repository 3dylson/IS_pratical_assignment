from itertools import chain

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data import DataLoader

from FashionCNN import FashionCNN
from FashionDataset import FashionDataset


def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


model = FashionCNN()
error = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if not os.path.exists("datasets/fashion-mnist_train.csv"):
    train_csv = pd.read_csv("fashion-mnist_train.csv")
    test_csv = pd.read_csv("fashion-mnist_test.csv")
else:
    train_csv = pd.read_csv("datasets/fashion-mnist_train.csv")
    test_csv = pd.read_csv("datasets/fashion-mnist_test.csv")
train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(train_set, batch_size=100)

class Model:
    def __init__(self):
        self.num_epochs = 5 # TODO: Change to 5
        # Lists for visualization of loss and accuracy
        self.loss_list = []
        self.iteration_list = []
        self.accuracy_list = []

        # Lists for knowing class-wise accuracy
        self.predictions_list = []
        self.labels_list = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: Check if model is trained if not train it

        # Transform data into Tensor that has a range from 0 to 1

        # a = next(iter(self.train_loader))
        # a[0].size()

        print(len(train_set))
        model.to(self.device)


        print("-- Model initialized --")
        print("Device: {}".format(self.device))
        print(model)

    def train(self):
        print("-- Training started...")
        count = 0
        for epoch in range(self.num_epochs):
            for images, labels in train_loader:
                # Transferring images and labels to GPU if available
                images, labels = images.to(self.device), labels.to(self.device)

                train = Variable(images.view(100, 1, 28, 28))
                labels = Variable(labels)

                # Forward pass
                outputs = model(train)
                loss = error(outputs, labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

                count += 1

                # Testing the model

                if not (count % 50):  # It's same as "if count % 50 == 0"
                    total = 0
                    correct = 0

                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        self.labels_list.append(labels)

                        test = Variable(images.view(100, 1, 28, 28))

                        outputs = model(test)

                        predictions = torch.max(outputs, 1)[1].to(self.device)
                        self.predictions_list.append(predictions)
                        correct += (predictions == labels).sum()

                        total += len(labels)

                    accuracy = correct * 100 / total
                    self.loss_list.append(loss.data)
                    self.iteration_list.append(count)
                    self.accuracy_list.append(accuracy)

                if not (count % 500):
                    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

    def show_accuracy(self):
        class_correct = [0. for _ in range(10)]
        total_correct = [0. for _ in range(10)]

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                test = Variable(images)
                outputs = model(test)
                predicted = torch.max(outputs, 1)[1]
                c = (predicted == labels).squeeze()

                for i in range(100):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    total_correct[label] += 1

        for i in range(10):
            print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))

    def show_classifications(self):
        predictions_l = [self.predictions_list[i].tolist() for i in range(len(self.predictions_list))]
        labels_l = [self.labels_list[i].tolist() for i in range(len(self.labels_list))]
        predictions_l = list(chain.from_iterable(predictions_l))
        labels_l = list(chain.from_iterable(labels_l))

        confusion_matrix(labels_l, predictions_l)
        print("Classification report for CNN :\n%s\n"
              % (metrics.classification_report(labels_l, predictions_l)))
