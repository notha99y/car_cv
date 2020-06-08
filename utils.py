from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


def get_images(directory):
    """
    """
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
    _collection = []
    for ext in IMG_EXTENSIONS:
        print("Doing extension: ", ext)
        for elem in tqdm(directory.rglob("*" + ext)):
            _collection.append(elem)
    return _collection


def view_image(row, train_test):
    """ View image with bbox
    Arguments
    ---------
    - Row of the csv
    - Train or Test dataset
    """

    image_name, l, t, r, b, class_idx = row
    class_name = car_dict[class_idx]
    drawn_img = Image.open(
        Path("stanford_car")
        / "car_data"
        / train_test
        / class_name
        / image_name
    )
    bbox = ImageDraw.Draw(drawn_img)
    bbox.rectangle([l, t, r, b], outline="red", fill=None)
    drawn_img.show()


if __name__ == "__main__":
    # directory = Path('stanford_car')
    # collection = get_images(directory)
    car_dict = {}
    with open(Path("stanford_car") / "names.csv", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        car_dict[i + 1] = line.strip()

    train_df = pd.read_csv(Path("stanford_car") / "anno_train.csv")
    view_image(train_df.iloc[1], "train")
