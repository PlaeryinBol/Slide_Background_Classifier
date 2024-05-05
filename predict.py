from glob import glob

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import dataset
import models
import utils
import wandb

torch.set_printoptions(sci_mode=False)


def predict(model=None, apply_smoothing=True, save_df=False, log_results=True):
    predict_df = pd.DataFrame()
    predict_df['image'] = glob(config.TEST_IMGS_PATH + '/**/*.jpg', recursive=True)

    test_dataset = dataset.PhotoDataset(predict_df, transforms=utils.transform_valid(), give_label=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 4, num_workers=config.NUM_WORKERS,
                             shuffle=False, multiprocessing_context='fork')

    # if the model is not explicitly passed, load the best model according to the data from the config
    if model is None:
        model = models.ClassificationModel(config.MODEL_NAME, apply_smoothing).to(config.DEVICE)
        model.load_state_dict(torch.load(config.EXP_NAME + '_best.pth'))

    with torch.no_grad():
        model.eval()
        test_preds = []
        for _, img in enumerate(tqdm(test_loader, total=len(test_loader), desc='Test batches')):
            img = img.to(config.DEVICE).float()
            test_output = model(img)
            predict = torch.argmax(test_output, axis=1)

            # use softmax if necessary: change the positive class for samples with confidence below the threshold
            if apply_smoothing:
                confident_predict = test_output[:, 1] > config.SOFTMAX_THRESHOLD
                predict = torch.where(confident_predict, torch.tensor(1), torch.tensor(0)).tolist()

            test_preds.append(predict)

    test_preds = np.concatenate(test_preds)
    predict_df['label'] = test_preds

    # visualize a table with predict labels in wandb
    if log_results and config.WANDB_MODE == 'online':
        predict_df['image'] = list(map(wandb.Image, predict_df['image']))
        test_table = wandb.Table(data=predict_df)
        wandb.run.log({"test_table": test_table})

    # save a dataframe with test predicts, if necessary
    if save_df:
        predict_df.to_csv('./test_predict.tsv', sep='\t', index=False)


if __name__ == '__main__':
    predict(model=None, apply_smoothing=True, save_df=True, log_results=True)
