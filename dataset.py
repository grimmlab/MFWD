from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, sampler
import albumentations as a
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from skimage import io as skio
from skimage.transform import resize


class MFWDDataset(Dataset):
    def __init__(self, data_folder: Path, labels_idx, img_size: tuple, transforms):
        self.data_folder = data_folder
        self.img_size = img_size
        self.transforms = transforms
        self.labels_idx = labels_idx
        self.img_list = sorted(list(data_folder.glob("*/*")))  # load all images that are present
        # filter the images so they contain ony the ones in the class_map
        self.img_list = [entry for entry in self.img_list if(entry.parent.stem in self.labels_idx)]
        self.labels = [lbl.parent.stem for lbl in self.img_list]

        return

    def _load_images(self, img_path):
        img = skio.imread(str(img_path))
        img = resize(img, self.img_size, clip=True,
                     preserve_range=True, anti_aliasing=True)
        img = img.astype(np.uint8)
        return img

    def get_class_distribution(self):
        count_dict = {k: 0 for k, v in self.labels_idx.items()}
        for label in self.labels:
            count_dict[label] += 1
        return count_dict

    def calculate_label_weights(self):
        count_dict = self.get_class_distribution()
        # label_weights as reciprocal of the count
        label_weights = [1 / count_dict[label] for label in self.labels]
        # label weights need to be of the same size as self.labels
        assert len(self.labels) == len(
            label_weights), f"length of label_weights not the same to weights"
        return label_weights

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = img_path.parent.stem
        label_id = self.labels_idx[f"{label}"]
        img = self._load_images(img_path)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, label_id



def load_class_map(class_map_f: Path):
    with open(class_map_f, "r") as f:
        class_dict = {k: idx for idx, k in enumerate(f.read().strip().split("\n"))}
    return class_dict


def get_dataloaders(train_folder, val_folder, class_map_f, batch_size, img_size, n_workers):
    means = (0.3300823, 0.32133222, 0.14781028)
    stds = (0.20910876, 0.24810849, 0.08084098)
    labels = load_class_map(class_map_f)

    train_transform = a.Compose([
        a.HorizontalFlip(),
        a.VerticalFlip(),
        a.RandomRotate90(),
        a.Transpose(),
        a.Normalize(
            mean=means,
            std=stds,
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    val_transform = a.Compose([
        a.Normalize(
            mean=means,
            std=stds,
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    train_ds = MFWDDataset(data_folder=train_folder, labels_idx=labels,
                           img_size=img_size, transforms=train_transform)
    val_ds = MFWDDataset(data_folder=val_folder, labels_idx=labels,
                         img_size=img_size, transforms=val_transform)
    sample_weights = train_ds.calculate_label_weights()
    weighted_random_sampler = sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True)
    t_loader = DataLoader(train_ds, sampler=weighted_random_sampler,
                          batch_size=batch_size, num_workers=n_workers)
    v_loader = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, num_workers=n_workers)

    return t_loader, v_loader, labels
