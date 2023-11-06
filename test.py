from dataset import get_dataloaders
from utils import seed_all
from train import load_model, validate_batch
from pathlib import Path

test_folder = Path("/data/MFWD/patches/test")
class_map_f = Path("/data/MFWD/class_map.txt")
model_save_path = Path("./models")
model_fname = "3_efficientnet_b0"
bs = 512
img_size = (224, 224)
n_workers = 2
device = "cuda"
seed_all(42)

_, test_loader, labels = get_dataloaders(
    test_folder, test_folder, class_map_f, batch_size=bs, img_size=img_size, n_workers=n_workers)
model = load_model(model_save_path, model_fname)
model = model.to(device)

gts, preds, f1_s = validate_batch(test_loader, model, device)
print(f1_s)
