import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        image_dim=(320, 320),
        pre_crop_transform=None,
        post_crop_transform=None,
        regression=False,
        flow_rate=False,
        feed_rate=False,
        z_offset=False,
        hotend=False,
        per_img_normalisation=False,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform

        self.image_dim = image_dim

        self.use_flow_rate = flow_rate
        self.use_feed_rate = feed_rate
        self.use_z_offset = z_offset
        self.use_hotend = hotend

        self.per_img_normalisation = per_img_normalisation

        self.targets = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        self.targets = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.img_path[idx])
        
        dim = self.image_dim[0] / 2

        left = self.dataframe.nozzle_tip_x[idx] - dim
        top = self.dataframe.nozzle_tip_y[idx] - dim
        right = self.dataframe.nozzle_tip_x[idx] + dim
        bottom = self.dataframe.nozzle_tip_y[idx] + dim

        image = Image.open(img_name)
        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)
        image = image.crop((left, top, right, bottom))

        if self.per_img_normalisation:
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            image = tfms(image)
            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean, std)(image)
        else:
            if self.post_crop_transform:
                image = self.post_crop_transform(image)

        if self.use_flow_rate:
            flow_rate_class = int(self.dataframe.flow_rate_class[idx])
            self.targets.append(flow_rate_class)

        if self.use_feed_rate:
            feed_rate_class = int(self.dataframe.feed_rate_class[idx])
            self.targets.append(feed_rate_class)

        if self.use_z_offset:
            z_offset_class = int(self.dataframe.z_offset_class[idx])
            self.targets.append(z_offset_class)

        if self.use_hotend:
            hotend_class = int(self.dataframe.hotend_class[idx])
            self.targets.append(hotend_class)

        y = torch.tensor(self.targets, dtype=torch.long)
        sample = (image, y)
        return sample
