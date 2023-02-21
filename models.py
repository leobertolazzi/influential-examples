import torch

# Logistic regression model
class LR(torch.nn.Module):

    def __init__(self, channels):
        super(LR, self).__init__()
        self.channels = channels

        self.linear = torch.nn.Linear(32 * 32 * channels, 10)   

    def forward(self, x):
        x = x.view(-1, 32 * 32 * self.channels)
        x = torch.sigmoid(self.linear(x))
        return x

# Multilayer perceptron with two hidden layer + dropout layers
class MLP(torch.nn.Module):

    def __init__(self, channels):
        super(MLP, self).__init__()
        self.channels = channels

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * channels, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 84),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(84, 10)
        ) 

    def forward(self, x):
        x = x.view(-1, 32 * 32 * self.channels)
        x = self.classifier(x)
        return x

# Convolutional neural network + dropout layers
class CNN(torch.nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            # Conv 1
            torch.nn.Conv2d(channels, 6, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            # Conv 2
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.3),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.3),
            torch.nn.Linear(84, 10)
        )
    
    def forward(self, x):
        
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x




