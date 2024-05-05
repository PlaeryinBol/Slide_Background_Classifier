from torch.utils.data import DataLoader, Dataset

import config
import utils


class PhotoDataset(Dataset):
    def __init__(self, df, transforms=None, give_label=True):
        """Performed only once when the Dataset object is instantiated.
        give_label should be False for test data"""
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.give_label = give_label

        if self.give_label:
            self.labels = self.df['label'].values

    def __len__(self):
        """Function to return the number of records in the dataset"""
        return self.df.shape[0]

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""

        # get labels
        if self.give_label:
            target = self.labels[index]

        # Load images
        img = utils.load_img(f'{self.df.loc[index]["image"]}')

        # Transform images
        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.give_label:
            return img, target
        else:
            return img


def create_dataloader(df, trn_idx, val_idx):
    """Creating train_loader and valid_loader"""
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)
    train_dataset = PhotoDataset(train_, transforms=utils.transform_train())
    valid_dataset = PhotoDataset(valid_, transforms=utils.transform_valid())
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                              shuffle=True, multiprocessing_context='fork')
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                              shuffle=False, multiprocessing_context='fork')

    return train_loader, valid_loader
