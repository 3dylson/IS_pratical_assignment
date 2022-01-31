from src.Model import Model

if __name__ == '__main__':
    model = Model()
    model.train()
    model.showAccuracy()
    model.showClassifications()
