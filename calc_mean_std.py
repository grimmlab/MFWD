from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as a
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from skimage import io as skio
from skimage.transform import resize
from tqdm import tqdm

data_folder = Path("/data/MFWD/patches/train")
bs = 5000
class MFWDDataset(Dataset):
    def __init__(self, data_folder:Path, labels, img_size:tuple, transforms):
        self.data_folder = data_folder
        self.img_size = img_size
        self.transforms = transforms
        self.img_list = sorted(list(data_folder.glob("*/*")))
        self.labels = labels
        return

    def _load_images(self, img_path):
        img = skio.imread(str(img_path))
        img = resize(img, self.img_size, clip=True, preserve_range=True, anti_aliasing=True)
        img = img.astype(np.uint8)
        return img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = img_path.parent.stem
        label_id = self.labels[f"{label}"]
        img = self._load_images(img_path)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img.type(torch.float32), label_id


train_transform = a.Compose([
    ToTensorV2(),
])

labels = {"ACHMI": 0,
          "AETCY": 1,
          "AGRRE": 2,
          "ALOMY": 3,
          "ARTVU": 4,
          "CHEAL": 5,
          "CIRAR": 6,
          "CONAR": 7,
          "ECHCG": 8,
          "GALAP": 9,
          "GASPA": 10,
          "GERMO": 11,
          "LAMAL": 12,
          "MATCH": 13,
          "PLAMA": 14,
          "POAAN": 15,
          "POLAM": 16,
          "POLCO": 17,
          "POROL": 18,
          "PULDY": 19,
          "SOLNI": 20,
          "SORVU": 21,
          "SSYOF": 22,
          "STEME": 23,
          "THLAR": 24,
          "VEROF": 25,
          "VIOAR": 26}

ds = MFWDDataset(data_folder=data_folder, labels=labels, img_size=(224,224), transforms=train_transform)
loader = DataLoader(ds, batch_size=bs, shuffle=False)
mean = 0.
std = 0.
nb_samples = 0.
tqdm_l = tqdm(loader, total = len(ds)//bs + 1)
for data, lbls in tqdm_l:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print(mean.numpy()/255.0)
print(std.numpy()/255.0)