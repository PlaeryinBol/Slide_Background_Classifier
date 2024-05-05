import os
import random
from glob import glob

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

import config
import wandb


def seed_everything():
    """Seeds the random number generators ensure reproducibility."""
    random.seed(config.SEED)
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_img(path, resize_to=None):
    """Image loading with optional resize."""
    img_bgr = cv2.imread(path)
    img_rgb = img_bgr[:, :, ::-1]

    # resize to the specified size
    if resize_to is not None:
        img_rgb = cv2.resize(img_rgb, resize_to)

    return img_rgb


def create_gallery(array, ncols=3):
    """Display multiple images in a gallery style."""
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows * ncols
    result = (array.reshape(nrows, ncols, height, width, intensity).swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def verify_data(df, samples_per_cls=4, resize_to=(512, 341), hist_threshold=0.25):
    """Verifies the data by visualizing random samples and analyzing pixel histograms for potential outlier images."""
    random_good_samples = df.loc[df['label'] == 1].sample(n=samples_per_cls)['image'].tolist()
    random_bad_samples = df.loc[df['label'] == 0].sample(n=samples_per_cls)['image'].tolist()
    random_samples = random_good_samples + random_bad_samples
    img_list = np.array([load_img(img_name, resize_to=resize_to) for img_name in random_samples])

    plt.figure(figsize=(10, 10))
    plt.imshow(create_gallery(img_list, ncols=samples_per_cls))
    plt.savefig('samples.png')

    pixel_count = resize_to[0] * resize_to[1]
    plt.figure(figsize=(10, 10))
    plt.ylim(0, pixel_count)
    for img_name in tqdm(df['image'], desc='Count pixel histogram'):
        img = load_img(img_name, resize_to=resize_to)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)

        # log those images whose pixel distribution looks suspicious - like there are too many pixels of the same color
        if np.any(hist > pixel_count * hist_threshold):
            config.LOG.warning(f'Possible outlier image: {img_name} - too many')

    plt.savefig('histogram.png')


def create_data_df():
    """Creates dataset dataframe."""
    # if the dataframe exists, read it from disk
    if os.path.exists(config.DF_PATH):
        df = pd.read_table(config.DF_PATH)
    else:
        good = glob(config.GOOD_BGS_PATH + '*/*.jpg')
        bad = glob(config.BAD_BGS_PATH + '*/*.jpg')
        # assign labels by folder
        labels = [1 if i < len(good) else 0 for i in range(len(good) + len(bad))]
        df = pd.DataFrame(data={'image': good + bad, 'label': labels})
        df.to_csv(config.DF_PATH, sep='\t', index=False)

        # saving the dataset in wandb
        data_table = wandb.Table(dataframe=df)
        data_table_artifact = wandb.Artifact("data_artifact", type="dataset")
        data_table_artifact.add(data_table, "data_table")
        data_table_artifact.add_file(config.DF_PATH)
        wandb.run.log({"data": data_table})
        wandb.run.log_artifact(data_table_artifact)

        # if necessary, run an exploratory data analysis
        if config.RUN_EDA:
            verify_data(df)

    config.LOG.info(f'Labels distribution:\n{df["label"].value_counts()}')
    return df


def transform_train(kit='first'):
    """Train augmentations."""
    if kit == 'first':
        transforms = [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE, p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5, rotate_limit=45),
            A.CoarseDropout(p=0.5),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    elif kit == 'second':
        transforms = [
            A.RandomResizedCrop(config.IMAGE_SIZE, config.IMAGE_SIZE, scale=(0.5, 1.0), p=1),
            A.ShiftScaleRotate(p=0.5, rotate_limit=45),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    else:
        raise ValueError('Wrong kit specification!')
    return A.Compose(transforms)


def transform_valid():
    """Val augmentations."""
    transform = [
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE, p=1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0)
    ]
    return A.Compose(transform)


def make_grid(batch, max_count=16):
    """Create a collage grid from a batch of images."""
    img_batch = np.copy(batch[:max_count].numpy())
    count, channels, width, height = img_batch.shape
    collage_size = int(np.ceil(np.sqrt(count)))
    collage = np.zeros((collage_size * width, collage_size * height, channels), dtype=np.uint8)

    for i in range(count):
        row = i // collage_size
        col = i % collage_size
        img = img_batch[i].transpose((1, 2, 0)).astype(np.uint8)
        img = cv2.resize(img, (width, height))
        collage[row * height:(row + 1) * height, col * width:(col + 1) * width] = img

    return Image.fromarray(collage)
