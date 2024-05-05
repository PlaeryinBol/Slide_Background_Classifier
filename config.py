import logging

import wandb

logging.basicConfig(filename="app.log", level=logging.INFO, filemode="w")
LOG = logging.getLogger()

EXP_NAME = 'efficientnet_50e_f75_512s_aug1_lr_1е-4_final'  # experiment name
WANDB_MODE = "online"  # disabled or online
APPLY_CROSS_VAL = True  # flag, whether to use cross-validation or pure train
FOLDS = 3  # number of folds for cross-validation
MODEL_NAME = 'efficientnet'  # model
IMAGE_SIZE = 512  # image size
FOCAL_LOSS_ALPHA = 0.75  # loss coefficient proportional to the frequency of the positive class in the dataset
EPOCHS = 25  # number of epochs
BATCH_SIZE = 24  # batch size
LR = 0.0001  # learning rate
GOOD_BGS_PATH = '/data/photo_bgs_dataset/good'  # folder with good backgrounds
BAD_BGS_PATH = '/data/photo_bgs_dataset/bad'  # folder with bad backgrounds
TEST_IMGS_PATH = '/data/photo_bgs_dataset/test'  # folder with test images
DF_PATH = './data_df.tsv'  # dataframe with dataset labels
NUM_CLASSES = 2  # number of classes
RUN_EDA = True  # flag, whether to perform exploratory data analysis
DEVICE = 'cuda'  # device
SEED = 42  # seed
NUM_WORKERS = 4  # number of dataloader workers
PREDICT_AFTER_TRAIN = True  # flag, whether to perform a test run immediately after training
SOFTMAX_THRESHOLD = 0.95  # threshold for canceling the positive class for samples with confidence below the threshold

run = wandb.init(
    mode=WANDB_MODE,
    project="photo_classifier",
    name=EXP_NAME if not APPLY_CROSS_VAL else EXP_NAME + '_cross_val',
    config={
        "image_size": IMAGE_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "dataset_id": "11.10.2023"
    })
