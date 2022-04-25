import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier

from train_config import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--seed", default=1234, type=int, help="Set seed for training"
)
parser.add_argument(
    "-e",
    "--epochs",
    default=MAX_EPOCHS,
    type=int,
    help="Number of epochs to train the model for",
)

args = parser.parse_args()
seed = args.seed

set_seed(seed)
logs_dir = "logs/logs-{}/{}/".format(DATE, seed)
logs_dir_default = os.path.join(logs_dir, "default")

make_dirs(logs_dir)
make_dirs(logs_dir_default)

tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/{}/{}/".format(DATE, seed),
    filename="MHResAttNet-{}-{}-".format(DATASET_NAME, DATE)
    + "{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
    save_top_k=3,
    mode="min",
)

model = ParametersClassifier(
    num_classes=3,
    lr=INITIAL_LR,
    gpus=NUM_GPUS,
    transfer=False,
)

data = ParametersDataModule(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    csv_file=DATA_CSV,
    dataset_name=DATASET_NAME,
    mean=DATASET_MEAN,
    std=DATASET_STD,
)

trainer = pl.Trainer(
    num_nodes=NUM_NODES,
    gpus=NUM_GPUS,
    distributed_backend=ACCELERATOR,
    max_epochs=args.epochs,
    logger=tb_logger,
    weights_summary=None,
    precision=16,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, data)
