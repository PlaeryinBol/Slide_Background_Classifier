import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

import config
import dataset
import models
import utils
import wandb

np.set_printoptions(suppress=True)


def cross_validation(df):
    folds = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED) \
        .split(np.arange(df.shape[0]), df['label'].to_numpy())

    best_f1 = 0
    for fold_idx, (trn_idx, val_idx) in enumerate(tqdm(folds, desc='Folds', total=config.FOLDS)):  # folds--------------
        train_loader, valid_loader = dataset.create_dataloader(df, trn_idx, val_idx)
        model = models.ClassificationModel(config.MODEL_NAME).to(config.DEVICE)
        # loss_fn = nn.BCEWithLogitsLoss().to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LR)

        for epoch in tqdm(range(config.EPOCHS), leave=False, desc='Epochs'):  # epochs----------------------------------
            model.train()
            train_loss = 0.0
            targets_list, predicts_list = [], []
            for batch_idx, (img_batch, lbl) in enumerate(tqdm(train_loader, total=len(train_loader),
                                                              desc='Train batches')):  # train--------------------------
                img_batch = img_batch.permute(0, 3, 1, 2)

                # visualize the first batch of images with wandb
                if batch_idx == 0:
                    grid = utils.make_grid(img_batch)
                    wandb.log({f"batches/train_batch_fold_{fold_idx}":
                               wandb.Image(grid, caption=f"first_batch_{fold_idx}.jpg")})

                img_batch = img_batch.to(config.DEVICE).float()
                label = lbl.to(config.DEVICE).long()

                gt_labels_one_hot = torch.zeros(label.size(0), config.NUM_CLASSES).to(config.DEVICE)
                gt_labels_one_hot.scatter_(1, label.unsqueeze(1), 1)

                output = model(img_batch)
                # loss = loss_fn(output, gt_labels_one_hot)
                loss = sigmoid_focal_loss(output, gt_labels_one_hot, alpha=config.FOCAL_LOSS_ALPHA, reduction='sum')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                target = lbl.cpu().tolist()
                predict = output.argmax(dim=1).cpu().tolist()
                targets_list.extend(target)
                predicts_list.extend(predict)

            epoch_loss = train_loss / len(train_loader)
            epoch_f1 = f1_score(targets_list, predicts_list, average='binary', zero_division=1.0)
            epoch_precision = precision_score(targets_list, predicts_list, average='binary', zero_division=1.0)
            epoch_recall = recall_score(targets_list, predicts_list, average='binary', zero_division=1.0)

            wandb.log({f"train_f1/fold-{fold_idx}": epoch_f1,
                       f"train_precision/fold-{fold_idx}": epoch_precision,
                       f"train_recall/fold-{fold_idx}": epoch_recall,
                       f"train_loss/fold-{fold_idx}": epoch_loss})
            # end train-------------------------------------------------------------------------------------------------

            with torch.no_grad():  # validation-------------------------------------------------------------------------
                model.eval()
                val_loss = 0.0
                targets_list_val, predicts_list_val = [], []
                for batch_idx, (img_batch, val_lbl) in enumerate(tqdm(valid_loader, total=len(valid_loader),
                                                                 desc='Valid batches')):
                    # img_batch = img_batch.permute(0, 3, 1, 2)

                    img_batch = img_batch.to(config.DEVICE).float()
                    label = val_lbl.to(config.DEVICE).long()

                    gt_labels_one_hot = torch.zeros(label.size(0), config.NUM_CLASSES).to(config.DEVICE)
                    gt_labels_one_hot.scatter_(1, label.unsqueeze(1), 1)

                    val_output = model(img_batch)
                    # val_loss = loss_fn(val_output, gt_labels_one_hot)
                    val_loss = sigmoid_focal_loss(val_output, gt_labels_one_hot,
                                                  alpha=config.FOCAL_LOSS_ALPHA, reduction='sum')

                    val_loss += val_loss
                    val_target = val_lbl.cpu().tolist()
                    val_predict = val_output.argmax(dim=1).cpu().tolist()
                    targets_list_val.extend(val_target)
                    predicts_list_val.extend(val_predict)

            epoch_loss_val = val_loss / len(valid_loader)
            epoch_f1_val = f1_score(targets_list_val, predicts_list_val, average='binary', zero_division=1.0)
            epoch_precision_val = precision_score(targets_list_val, predicts_list_val, average='binary', zero_division=1.0)
            epoch_recall_val = recall_score(targets_list_val, predicts_list_val, average='binary', zero_division=1.0)

            wandb.log({"epochs/epoch": epoch,
                       f"val_f1/fold-{fold_idx}": epoch_f1_val,
                       f"val_precision/fold-{fold_idx}": epoch_precision_val,
                       f"val_recall/fold-{fold_idx}": epoch_recall_val,
                       f"val_loss/fold-{fold_idx}": epoch_loss_val})

            # tracking the best f1 value between folds
            if epoch_f1_val > best_f1:
                best_f1 = epoch_f1_val

            wandb.run.summary["best_val_f1"] = best_f1
            # end validation & epoch------------------------------------------------------------------------------------

        del model, optimizer, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()


def train(df):
    train_dataset = dataset.PhotoDataset(df, transforms=utils.transform_train())
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                              shuffle=True, multiprocessing_context='fork')

    model = models.ClassificationModel(config.MODEL_NAME).to(config.DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_f1 = 0
    best_loss = torch.inf
    for epoch in tqdm(range(config.EPOCHS), leave=False, desc='Epochs'):
        model.train()
        running_loss = 0.0
        targets_list, predicts_list = [], []
        for batch_idx, (img_batch, lbl) in enumerate(tqdm(train_loader, total=len(train_loader), desc='Train batches')):

            img_batch = img_batch.permute(0, 3, 1, 2)

            # if batch_idx == 0:
            #     grid = utils.make_grid(img_batch)
            #     wandb.log({"first_train_batch": wandb.Image(grid, caption="first_train_batch.jpg")})

            img_batch = img_batch.to(config.DEVICE).float()
            label = lbl.to(config.DEVICE).long()
            gt_labels_one_hot = torch.zeros(label.size(0), config.NUM_CLASSES).to(config.DEVICE)
            gt_labels_one_hot.scatter_(1, label.unsqueeze(1), 1)

            output = model(img_batch)
            # loss = loss_fn(output, label)
            loss = sigmoid_focal_loss(output, gt_labels_one_hot, alpha=config.FOCAL_LOSS_ALPHA, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            target = lbl.cpu().tolist()
            predict = output.argmax(dim=1).cpu().tolist()
            targets_list.extend(target)
            predicts_list.extend(predict)

        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = f1_score(targets_list, predicts_list, average='binary', zero_division=1.0)
        epoch_precision = precision_score(targets_list, predicts_list, average='binary', zero_division=1.0)
        epoch_recall = recall_score(targets_list, predicts_list, average='binary', zero_division=1.0)

        wandb.log({"epochs/epoch": epoch,
                   "train_f1": epoch_f1,
                   "train_precision": epoch_precision,
                   "train_recall": epoch_recall,
                   "train_loss": epoch_loss})

        # saving the model with maximum f1 and minimum loss to disk
        if (epoch_f1 > best_f1) and (epoch_loss < best_loss):
            best_f1 = epoch_f1
            best_loss = epoch_loss
            torch.save(model.state_dict(), config.EXP_NAME + '_best.pth')

    # if need to run a test run
    if config.PREDICT_AFTER_TRAIN:
        predict(model, save=True, log_results=True)

    # save the model on the last epoch, log the results
    torch.save(model.state_dict(), config.EXP_NAME + '_last.pth')
    wandb.run.summary["best_train_f1"] = best_f1
    best_model = wandb.Artifact(f'{config.EXP_NAME}_{wandb.run.id}', type="model")
    best_model.add_file(config.EXP_NAME + '_best.pth')
    wandb.run.log_artifact(best_model, aliases=["latest"])
    del model, optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    utils.seed_everything()
    df = utils.create_data_df()

    # either cross-validation or standard train
    if config.APPLY_CROSS_VAL:
        cross_validation(df)
    else:
        train(df)
