import torch

from Model import Model
from os.path import exists

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
