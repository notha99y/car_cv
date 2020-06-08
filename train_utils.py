from pathlib import Path

import torch
from dataset import StanfordCarDataset
from torchvision import transforms


def save_ckpt(model, optimizer, epoch, losses, accuracies, path):
    """Save model checkpoints
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": losses["train"],
            "train_acc": accuracies["train"],
            "test_loss": losses["test"],
            "test_acc": accuracies["test"],
        },
        path,
    )
    print("Checkpoint save at: ", path)


def load_ckpt(ckpt_path, model, optimizer):
    """Load model check points
    """

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    losses = {"train_loss": ckpt["train_loss"], "test_loss": ckpt["test_loss"]}
    accuracies = {"train_acc": ckpt["train_acc"], "test_acc": ckpt["test_acc"]}

    return model, optimizer, epoch, losses, accuracies


def get_data_transforms():
    """Get data tranformation"""

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    return data_transforms


def get_data_sets(train_config, data_transforms):
    """Get datasets"""
    contexts = ["train", "test"]

    if train_config["dataset"]["name"] == "StanfordCarDataset":
        data_path = Path(".") / "stanford_car"
        names_csv = data_path / "names.csv"

        anno_csvs = {x: data_path / "anno_{}.csv".format(x) for x in contexts}
        root_dirs = {x: data_path / "car_data" / x for x in contexts}

        datasets = {
            x: StanfordCarDataset(
                names_csv,
                anno_csvs[x],
                root_dirs[x],
                crop=False,
                transform=data_transforms[x],
            )
            for x in contexts
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(
                datasets[x], batch_size=4, shuffle=True, num_workers=4
            )
            for x in contexts
        }

        class_names = datasets["train"].car_names

    else:
        print("No other datasets")

    return datasets, dataloaders, class_names
