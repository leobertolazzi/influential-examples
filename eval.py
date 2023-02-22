import os
import glob
import argparse
import warnings
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
from captum.influence import TracInCPFast

from trainer import Trainer, get_data
from models import LR, MLP, CNN


class Evaluation():
    def __init__(
        self,
        training_data,
        validation_data,
        test_data,
        dataset_type='fashion-mnist',
        ):

        self.dataset_type = dataset_type
        
        self.epochs = 10 if dataset_type == 'fashion-mnist' else 20
        self.save_every = 1 if dataset_type == 'fashion-mnist' else 2
        self.channels = 1 if dataset_type == 'fashion-mnist' else 3

        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.inverse_normalize = torchvision.transforms.Compose([ 
            torchvision.transforms.Normalize(mean = [0], std = [1/0.5]),
            torchvision.transforms.Normalize(mean = [-0.5], std = [1.]), 
        ])

        self.imshow_transform = lambda tensor_in_dataset: self.inverse_normalize(tensor_in_dataset.squeeze()).permute(1, 2, 0)

        self.label_to_class = (
            't-shirt/top',
            'trouser',
            'pullover',
            'dress',
            'coat',
            'sandal',
            'shirt',
            'sneaker',
            'bag',
            'ankle boot'
            ) if dataset_type == 'fashion-mnist' else (
                'airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
            )

    def add_train_label_noise(self, percentage=10):

        noisy_data = [list(sample) for sample in self.training_data]
        
        label_ids = np.random.choice(len(self.training_data), size=round(len(self.training_data)*(percentage/100)), replace=False)
        rand_labels = np.random.choice(10, size=round(len(self.training_data)*(percentage/100)), replace=True)

        for n, id in np.ndenumerate(label_ids):
            if noisy_data[id][1] != rand_labels[n]:
                noisy_data[id][1] = rand_labels[n]
            else:
                filtered_labels = list(range(10))
                filtered_labels.remove(rand_labels[n])
                noisy_data[id][1] = np.random.choice(filtered_labels, size=1)[0]

        noisy_data = [tuple(sample) for sample in noisy_data]

        return noisy_data
    
    def reinitialize_model(self, model_type):

        if model_type == 'lr':
            model = LR(self.channels)
        elif model_type == 'mlp':
            model = MLP(self.channels)
        elif model_type == 'cnn':
            model = CNN(self.channels)

        return model

    def checkpoints_load_func(self, model, path):

        weights = torch.load(path)
        model.load_state_dict(weights["model_state_dict"])
        
        return 1.

    def process_test_examples(self, model, test_examples_indices):

        test_examples_features = torch.stack([test_data[i][0] for i in test_examples_indices])
        test_examples_predicted_probs, test_examples_predicted_labels = torch.max(
            torch.nn.functional.softmax(model(test_examples_features), dim=1), dim=1)
        test_examples_true_labels = torch.Tensor([test_data[i][1] for i in test_examples_indices]).long()

        return test_examples_features, test_examples_predicted_probs, test_examples_predicted_labels, test_examples_true_labels
    
    def percentage_noisy_influential_examples(
        self,
        model, 
        training_data,
        noisy_examples_indices,
        test_examples_indices, 
        checkpoints_dir, 
        checkpoints_load_func
        ):

        test_examples_features, test_examples_predicted_probs, test_examples_predicted_labels, test_examples_true_labels = self.process_test_examples(
            model,
            test_examples_indices
        )

        proponents_indices, proponents_influence_scores, opponents_indices, opponents_influence_scores = self.compute_influence_scores(
            model,
            training_data,
            test_examples_features,
            test_examples_true_labels,
            checkpoints_dir,
            checkpoints_load_func,
        )

        total_noisy_examples = 0
        total_noisy_examples += sum(np.count_nonzero(proponents_indices.flatten() == noisy_example) for noisy_example in noisy_examples_indices)
        total_noisy_examples += sum(np.count_nonzero(opponents_indices.flatten() == noisy_example) for noisy_example in noisy_examples_indices)

        noisy_examples_percentage = total_noisy_examples/(len(proponents_indices.flatten())*2)

        return noisy_examples_percentage
    
    def train_and_compute_noisy_percentage(
        self,
        model,
        model_type,
        noisy_training_data,
        noisy_examples_indices,
        test_examples_indices,
        device=None
        ):

        trainer = Trainer(
            model,
            model_type=model_type,
            dataset_type=self.dataset_type,
            device=device
        )

        checkpoints_dir = trainer.fit(
            noisy_training_data,
            self.validation_data,
            self.test_data,  
            epochs=self.epochs,
            save_every=self.save_every,
            show_training=False
        )

        model = self.reinitialize_model(model_type)

        noisy_examples_percentage = self.percentage_noisy_influential_examples(
                model, 
                noisy_training_data,
                noisy_examples_indices,
                test_examples_indices, 
                checkpoints_dir, 
                self.checkpoints_load_func
            )
        
        return noisy_examples_percentage

    def compute_self_influence(
            self,
            model,
            training_data,
            checkpoints_dir,
            checkpoints_load_func,
        ):

        tracin_cp_fast = TracInCPFast(
            model=model,
            final_fc_layer=list(model.children())[-1],
            train_dataset=training_data,
            checkpoints=checkpoints_dir,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            batch_size=2048,
        )

        self_influence_scores = tracin_cp_fast.self_influence()

        return self_influence_scores
    
    def compute_influence_scores(
        self,
        model,
        training_data,
        test_examples_features,
        test_examples_true_labels,
        checkpoints_dir,
        checkpoints_load_func,
        k=10
        ):

        tracin_cp_fast = TracInCPFast(
            model=model,
            final_fc_layer=list(model.children())[-1],
            train_dataset=training_data,
            checkpoints=checkpoints_dir,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            batch_size=2048,
            vectorize=False,
        )

        proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
            (test_examples_features, test_examples_true_labels),
            k=k,
            proponents=True
        )
        opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
            (test_examples_features, test_examples_true_labels),
            k=k,
            proponents=False
        )

        return proponents_indices, proponents_influence_scores, opponents_indices, opponents_influence_scores
    
    def display_test_example(
        self,
        example,
        true_label,
        predicted_label, 
        predicted_prob,
        label_to_class,
        figsize=(6,6),
        ):

        fig, ax = plt.subplots(figsize=figsize)
        if self.dataset_type == 'fashion-mnist':
            ax.imshow(torch.clip(self.inverse_normalize(example).squeeze(), 0, 1))
        else:
            ax.imshow(torch.clip(self.imshow_transform(example), 0, 1))
        fig.suptitle(f"True class: {label_to_class[true_label]}" +  "\n" + 
            f"Predicted class: {label_to_class[predicted_label]}" + "\n" + f"Predicted prob: {predicted_prob:.3f}")
        plt.show()

    def display_training_examples(
        self,
        examples,
        true_labels,
        label_to_class,
        figsize=(10,4),
        proponents=True
        ):

        title = 'Proponents' if proponents == True else 'Opponents'

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        num_examples = len(examples)
        for i in range(num_examples):
            ax = fig.add_subplot(1, num_examples, i+1)
            if self.dataset_type == 'fashion-mnist':
                ax.imshow(torch.clip(self.inverse_normalize(examples[i]).squeeze(), 0, 1))
            else:
                ax.imshow(torch.clip(self.imshow_transform(examples[i]), 0, 1))
            ax.set_title(label_to_class[true_labels[i]])
        plt.show()

    def display_proponents_and_opponents(
        self,
        test_examples_batch,
        proponents_indices,
        opponents_indices,
        test_examples_true_labels,
        test_examples_predicted_labels,
        test_examples_predicted_probs
        ):

        for (
            test_example,
            test_example_proponents,
            test_example_opponents,
            test_example_true_label,
            test_example_predicted_label,
            test_example_predicted_prob,
        ) in zip(
            test_examples_batch,
            proponents_indices,
            opponents_indices,
            test_examples_true_labels,
            test_examples_predicted_labels,
            test_examples_predicted_probs,
        ):

            self.display_test_example(
                test_example,
                test_example_true_label,
                test_example_predicted_label,
                test_example_predicted_prob,
                self.label_to_class,
            )

            test_example_proponents_tensors, test_example_proponents_labels = zip(
                *[self.training_data[i] for i in test_example_proponents]
            )
            self.display_training_examples(
                test_example_proponents_tensors, test_example_proponents_labels, self.label_to_class, figsize=(15, 3)
            )

            test_example_opponents_tensors, test_example_opponents_labels = zip(
                *[self.training_data[i] for i in test_example_opponents]
            )
            self.display_training_examples(
                test_example_opponents_tensors, test_example_opponents_labels, self.label_to_class, figsize=(15, 3), proponents=False
            )

    def display_self_influence_results(
        self,
        self_influence_scores,
        is_noisy,
        model_type='lr'
        ):

        if not os.path.exists(os.path.join('results',self.dataset_type)):
            os.makedirs(os.path.join('results',self.dataset_type))

        auc_score = roc_auc_score(is_noisy, self_influence_scores)
        fpr, tpr, _ = roc_curve(is_noisy, self_influence_scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        fontsize = 10
        ax.set_ylabel("TPR (proportion of mislabelled examples found)", fontsize=fontsize)
        ax.set_xlabel("FPR (proportion of correctly-labelled examples examined)", fontsize=fontsize)
        ax.set_title(f"ROC curve - {model_type.upper()}" + "\n" + f"AUC score: {auc_score:.3f}")
        path = os.path.join('results', self.dataset_type, f'roc_{model_type}')
        plt.savefig(path, dpi=200)
        plt.show()

    def display_label_noise(
        self,
        train_noise_percentages,
        model_noisy_examples_percentages
        ):

        if not os.path.exists(os.path.join('results',self.dataset_type)):
            os.makedirs(os.path.join('results',self.dataset_type))

        x_axis = np.arange(len(train_noise_percentages))

        if self.model_type == 'lr':
            color = 'tab:red'
        elif self.model_type == 'mlp':
            color = 'tab:orange'
        elif self.model_type == 'cnn':
            color = 'tab:blue'
            
        bars = plt.bar(x_axis, model_noisy_examples_percentages, 0.4, color=color)
        for bar in bars:
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", ha='center')
        plt.xticks(x_axis, train_noise_percentages)
        plt.xlabel(f"Label noise percentage in the training set")
        plt.ylabel("Percentage of noisy influential examples")
        plt.title(f"Model: {self.model_type.upper()}" + "\n" + "Noisy influential examples vs. amount of noise in training data")
        path = os.path.join('results', self.dataset_type, f'label_noise_{self.model_type}.png')
        plt.savefig(path, dpi=200)
        plt.show()

    def sample_influence(
        self,
        model,
        model_type='lr',
        device=None
        ):

        checkpoints_dir = os.path.join('checkpoints', self.dataset_type, model_type)
        if not os.path.exists(checkpoints_dir):
            trainer = Trainer(
                model,
                model_type=model_type,
                dataset_type=self.dataset_type,
                device=device
                )
            
            checkpoints_dir = trainer.fit(
                self.training_data,
                self.validation_data,
                self.test_data,  
                epochs=self.epochs,
                save_every=self.save_every
            )
        else:
            checkpoints_dir = glob.glob(os.path.join(checkpoints_dir, "*.pt"))

        test_examples_indices = np.random.choice(len(self.test_data), size=1, replace=False)

        model = self.reinitialize_model(model_type)

        test_examples_features, test_examples_predicted_probs, test_examples_predicted_labels, test_examples_true_labels = self.process_test_examples(
            model,
            test_examples_indices
            )
        
        proponents_indices, proponents_influence_scores, opponents_indices, opponents_influence_scores = self.compute_influence_scores(
            model,
            self.training_data,
            test_examples_features,
            test_examples_true_labels,
            checkpoints_dir,
            self.checkpoints_load_func,
        )

        self.display_proponents_and_opponents(
            test_examples_features,
            proponents_indices,
            opponents_indices,
            test_examples_true_labels,
            test_examples_predicted_labels,
            test_examples_predicted_probs,
        )

        print("End of evaluation.")

    def self_influence(
        self,
        model,
        model_type='lr',
        device=None
        ):

        noisy_training_data = self.add_train_label_noise(percentage=10)

        noisy_labels = torch.Tensor([noisy_training_data[i][1] for i in range(len(noisy_training_data))])
        correct_labels = torch.Tensor([self.training_data[i][1] for i in range(len(self.training_data))])
        is_noisy = noisy_labels != correct_labels

        trainer = Trainer(
            model,
            model_type=model_type,
            dataset_type=self.dataset_type,
            device=device
            )
        
        checkpoints_dir = trainer.fit(
            noisy_training_data,
            self.validation_data,
            self.test_data,  
            epochs=self.epochs,
            save_every=self.save_every
        )

        model = self.reinitialize_model(model_type)
        
        self_influence = self.compute_self_influence(
            model,
            noisy_training_data,
            checkpoints_dir,
            self.checkpoints_load_func
            )
        
        self.display_self_influence_results(
            self_influence,
            is_noisy,
            model_type=model_type
            )
        
        print("End of evaluation.")
    
    def label_noise(
        self,
        model,
        model_type='lr',
        percentages=None,
        device=None
        ):

        self.model_type = model_type

        model_noisy_examples_percentages = []
        train_noise_percentages = []

        for percentage in percentages:

            print(f"Training model with {percentage}% of label noise in the training set...")

            noisy_training_data = self.add_train_label_noise(percentage=percentage)
            test_examples_indices = range(len(test_data))
            noisy_examples_indices = [i for i in range(len(noisy_training_data)) if noisy_training_data[i][1] != training_data[i][1]]

            model = self.reinitialize_model(self.model_type)
        
            model_noisy_examples_percentage = self.train_and_compute_noisy_percentage(
                model,
                model_type,
                noisy_training_data,
                noisy_examples_indices,
                test_examples_indices,
                device=device
            )

            model_noisy_examples_percentages.append(model_noisy_examples_percentage*100)

            train_noise_percentages.append(f"{percentage}%")

        print("End of evaluation.")

        self.display_label_noise(
            train_noise_percentages,
            model_noisy_examples_percentages
            )


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--dataset", type=str, default="fashion-mnist",
        choices=["fashion-mnist", "cifar10"])
    parser.add_argument("--model_type", type=str, default="lr",
        choices=["lr", "mlp", "cnn"])
    parser.add_argument("--eval_type", type=str, default="sample_influence",
        choices=["self_influence", "label_noise", "sample_influence"])
    parser.add_argument("--device", type=str, default=device,
        choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()
    
    training_data, validation_data, test_data = get_data(args.dataset)

    channels = 1 if args.dataset == 'fashion-mnist' else 3
    if args.model_type == 'mlp':
        model = MLP(channels)
    elif args.model_type == 'lr':
        model = LR(channels)
    elif args.model_type == 'cnn':
        model = CNN(channels)

    if args.eval_type == 'sample_influence':
        evaluation = Evaluation(
            training_data,
            validation_data,
            test_data,
            dataset_type = args.dataset
            )
        
        evaluation.sample_influence(
            model,
            model_type=args.model_type,
            device=args.device
            )

    elif args.eval_type == 'self_influence':
        evaluation = Evaluation(
            training_data,
            validation_data,
            test_data,
            dataset_type = args.dataset
            )
        
        evaluation.self_influence(
            model,
            model_type=args.model_type,
            device=args.device
            )
    
    elif args.eval_type == 'label_noise':
        evaluation = Evaluation(
            training_data,
            validation_data,
            test_data,
            dataset_type = args.dataset,
            )

        percentages = [5, 10, 20]
        
        evaluation.label_noise(
            model,
            model_type=args.model_type,
            percentages=percentages,
            device=args.device
            )

