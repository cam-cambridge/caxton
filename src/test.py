import os
import argparse
import pytorch_lightning as pl
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--seed", default=1234, type=int, help="Set seed for training"
)

args = parser.parse_args()
seed = args.seed

set_seed(seed)

model = ParametersClassifier.load_from_checkpoint(
    checkpoint_path=os.environ.get("CHECKPOINT_PATH"),
    num_classes=3,
    lr=INITIAL_LR,
    gpus=1,
    transfer=False,
)
model.eval()

data = ParametersDataModule(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    csv_file=DATA_CSV,
    image_dim=(320, 320),
    dataset_name=DATASET_NAME,
    mean=DATASET_MEAN,
    std=DATASET_STD,
    transform=False,
)
data.setup('test')

trainer = pl.Trainer(
    num_nodes=1,
    gpus=1,
    weights_summary=None,
    precision=16,
)

trainer.test(model, datamodule=data)
