from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class StanfordCarDataset(Dataset):
    """Stanford Car Dataset"""

    def __init__(
        self, names_csv, anno_csv, root_dir, crop=False, transform=None
    ):
        """
        Args:
            - names_csv (str): Path to the csv file with the names of the classes
            - anno_csv (str): Path to the csv file with the annotations (train/ test)
            - root_dir (str): Directory with all the images (train/ test)
            - crop (bool): To crop using bbox information
            - transform (callable, optional): Optional transform to be applied on a sample
        """

        car_dict = {}
        with open(names_csv, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            car_dict[i] = line.strip()
        self.car_dict = car_dict
        self.car_names = list(self.car_dict.values())
        self.df = pd.read_csv(anno_csv)
        self.root_dir = root_dir
        self.crop = crop
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, l, t, r, b, car_idx = self.df.iloc[idx]
        car_idx -= 1
        car_name = self.car_dict[car_idx]
        img_path = Path(self.root_dir) / car_name / image_name
        img = io.imread(img_path)
        if len(img.shape) != 3:
            img = np.stack((img,) * 3, axis=-1)
        if self.crop:
            img = img[l:r, t:b]

        if self.transform:
            img = self.transform(img)
        return img, car_idx


if __name__ == "__main__":
    contexts = ["train", "test"]
    data_path = Path(".") / "stanford_car"
    names_csv = data_path / "names.csv"

    anno_csvs = {x: data_path / "anno_{}.csv".format(x) for x in contexts}
    root_dirs = {x: data_path / "car_data" / x for x in contexts}

    data_transforms = {
        "train": transforms.Compose(
            [
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
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    datasets = {
        x: StanfordCarDataset(
            names_csv, anno_csvs[x], root_dirs[x], crop=False, transform=None,
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(len(datasets["train"]))):
        img, label = datasets["train"][i]
