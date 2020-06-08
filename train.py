"""
Training script for Fine Grain classification of Car
"""

import copy
import time
from pathlib import Path

from tqdm import tqdm

import mlflow
import mlflow.pytorch
import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from train_utils import *


def train_model(
    model, dataloaders, criterion, optimizer, scheduler, train_config, device,
):
    """Trains the model
    """

    num_epochs = train_config["training"]["epoch"]
    model_name = train_config['models']['name']
    dataset_name = dataloaders["train"].dataset.__class__.__name__

    tic = time.time()
    model = model.to(device)
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    contexts = ["train", "test"]
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in contexts}

    # mlflow stuffs
    host = train_config["mlflow"]["host"]
    port = train_config["mlflow"]["port"]
    mlflow.set_tracking_uri(f"http://{host}:{port}")
    mlflow.set_experiment(f"{model_name}_{dataset_name}")

    with mlflow.start_run():
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            losses = {}
            accuracies = {}
            # Each epoch has a training and testing phase (Note: for the stanford car dataset, the validation and test is synonymous)
            for phase in contexts:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for images, labels in tqdm(dataloaders[phase]):
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # Running Statistics (Running means within the epoch)
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

                # Epoch Statistics
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(
                    "{} phase. Loss: {:.2f} Acc: {:.2f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )
                losses[phase] = epoch_loss
                accuracies[phase] = epoch_acc.item()

                # deep copy and save out the model
                if phase == "test" and epoch_acc > best_acc:
                    save_path = (
                        Path(".")
                        / "checkpoints"
                        / "{}_{}_{:.2f}.pth".format(
                            model_name, epoch, epoch_loss
                        )
                    )
                    print("Saving best model")
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    save_ckpt(
                        model, optimizer, epoch, losses, accuracies, save_path
                    )
                    # Logging as mlflow artifactss
                    mlflow.pytorch.log_model(model, "models")

            mlflow.log_metrics(
                {
                    "train_acc": accuracies["train"],
                    "train_loss": losses["train"],
                    "test_acc": accuracies["test"],
                    "test_loss": losses["test"],
                },
                step=epoch,
            )

    time_elapsed = time.time() - tic
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best test Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)

    # Log the model as a mlflow artifact
    return model


if __name__ == "__main__":
    import toml

    train_config = toml.load(Path(".") / "configs" / "fg_train.toml")

    data_transforms = get_data_transforms()

    datasets, dataloaders, class_names = get_data_sets(
        train_config, data_transforms
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.get_resnets(
        num_out_classes=len(class_names), pretrained=True, train_config= train_config
    )
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config["optimizer"]["learning_rate"],
        momentum=train_config["optimizer"]["momentum"],
    )
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=train_config["lr_scheduler"]["step_size"],
        gamma=train_config["lr_scheduler"]["gamma"],
    )

    print("Starting Training")
    print("-" * 10)
    train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        exp_lr_scheduler,
        train_config,
        device,
    )
