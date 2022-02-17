import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

# Generating a noisy multi-sin wave

if __name__ == '__main__':
    if not exists("trainedModel.pth"):
        model = Model()
        model.train()
        model.show_accuracy()
        model.show_classifications()
        torch.save(model, 'trainedModel.pth')
    else:
        print("Model available... loading...")
        # Loading the saved model
        model = torch.load("trainedModel.pth")
        model.show_classifications()
