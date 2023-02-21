import os
import glob
import argparse
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from models import LR, MLP, CNN

def get_data(dataset='fashion-mnist'): 

    normalize = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if dataset == 'fashion-mnist':
        full_training_data = torchvision.datasets.FashionMNIST('data', train=True, transform=normalize, download=True) 
        test_data = torchvision.datasets.FashionMNIST('data', train=False, transform=normalize, download=True)
    elif dataset == 'cifar10':
        full_training_data = torchvision.datasets.CIFAR10('data', train=True, transform=normalize, download=True)
        test_data = torchvision.datasets.CIFAR10('data', train=False, transform=normalize, download=True)

    num_samples = len(full_training_data)
    training_samples = int(num_samples * 0.8 + 1)
    validation_samples = num_samples - training_samples

    training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples, validation_samples])

    return training_data, validation_data, test_data


class Trainer:
    def __init__(
        self,
        model,
        model_type='lr',
        dataset_type='fashion-mnist',
        device=None
        ):
        
        self.model = model.to(device)
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.device = device
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.000001, momentum=0.9)
        self.cost_function = torch.nn.CrossEntropyLoss()

    def fit(
        self,
        training_data,
        validation_data,
        test_data,  
        epochs=10,
        save_every=1,
        show_training=True
        ):
        
        self.save_every = save_every
        self.checkpoints_dir = os.path.join('checkpoints', self.dataset_type, self.model_type)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        train_loader = DataLoader(training_data, 128, shuffle=True)
        val_loader = DataLoader(validation_data, 128, shuffle=False)
        test_loader = DataLoader(test_data, 128, shuffle=False)

        test_loss, test_accuracy = self.test_step(test_loader)
        if show_training == True:
            print('Accuracy on the test set before training: {:.2f}\n'.format(test_accuracy))

        for current_epoch in range(epochs):

            train_loss, train_accuracy = self.training_step(train_loader, current_epoch)
            val_loss, val_accuracy = self.test_step(val_loader)

            if show_training == True:
                print('[{:d}/{:d}]\t loss/train: {:.5f}\t loss/val: {:.5f}\t acc/train: {:.2f}\t acc/val: {:.2f}'.format(
                    current_epoch+1, epochs, train_loss, val_loss, train_accuracy, val_accuracy)
                )

        checkpoints_dir = glob.glob(os.path.join(self.checkpoints_dir, "*.pt"))

        test_loss, test_accuracy = self.test_step(test_loader)
        if show_training == True:
            print('\nAccuracy on the test set after training: {:.2f}'.format(test_accuracy))

        return checkpoints_dir

    def training_step(self, data_loader, current_epoch):

        samples = 0.
        cumulative_loss = 0.
        cumulative_accuracy = 0.

        self.model.train() 

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()  
            outputs = self.model(inputs)
            loss = self.cost_function(outputs, targets)  
            loss.backward()  
            self.optimizer.step()  

            samples += inputs.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(dim=1)

            cumulative_accuracy += predicted.eq(targets).sum().item()

        if current_epoch % self.save_every == 0:
            checkpoint_name = "-".join(["checkpoint", str(current_epoch) + ".pt"])
            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": cumulative_loss/samples,
                },
            os.path.join(self.checkpoints_dir, checkpoint_name),
            )

        return cumulative_loss/samples, cumulative_accuracy/samples*100

    def test_step(self, data_loader):

        samples = 0.
        cumulative_loss = 0.
        cumulative_accuracy = 0.

        self.model.eval() 

        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(data_loader):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.cost_function(outputs, targets)   

                samples += inputs.shape[0]
                cumulative_loss += loss.item()
                _, predicted = outputs.max(dim=1)

                cumulative_accuracy += predicted.eq(targets).sum().item()

        return cumulative_loss/samples, cumulative_accuracy/samples*100


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--dataset", type=str, default="fashion-mnist",
        choices=["fashion-mnist", "cifar10"])
    parser.add_argument("--model_type", type=str, default="lr",
        choices=["lr", "mlp", "cnn"])
    parser.add_argument("--device", type=str, default=device,
        choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()
    
    epochs = 10 if args.dataset == 'fashion-mnist' else 20
    save_every = 1 if args.dataset == 'fashion-mnist' else 2
    channels = 1 if args.dataset == 'fashion-mnist' else 3

    training_data, validation_data, test_data = get_data(args.dataset)

    if args.model_type == 'mlp':
        model = MLP(channels)
    elif args.model_type == 'lr':
        model = LR(channels)
    elif args.model_type == 'cnn':
        model = CNN(channels)

    trainer = Trainer(
        model, 
        model_type=args.model_type,
        dataset_type=args.dataset,
        device=args.device
        )

    checkpoints_dir = trainer.fit(
        training_data,
        validation_data,
        test_data,  
        epochs=epochs,
        save_every=save_every
    )