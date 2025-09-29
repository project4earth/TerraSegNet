import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm
from models.unet import UNet
from models.deeplabv3plus import DeepLabv3Plus
from models.hrnet import HRNet
from models.segformer import SegFormer
from models.unet import UNet
from models.bisenetv1 import BiSeNetV1
from models.bisenetv2 import BiSeNetV2
from models.fast_scnn import Fast_SCNN
from models.seaformerpp import SeaFormerPP
from models.hrcloudnet import HRcloudNet
from models.cdnetv2 import CDNetV2
from models.unetformer import UNetFormer
from models.cmlformer import CMLFormer
from models.cmtfnet import CMTFNet
from models.terrasegnet import TerraSegNet
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)

def minmax_normalization(image):
    normalized = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        normalized[c] = image[c] / 255.0
    return normalized

class SatelliteDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith('.tiff') or f.endswith('.tif')
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with rasterio.open(img_path) as src:
                image = src.read()
                meta = src.meta
        except Exception as e:
            print(f"[ERROR] Gagal membaca {img_path}: {e}")
            return None
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = minmax_normalization(image)
        image_tensor = torch.tensor(image, dtype=torch.float32)
        return image_tensor, img_path, meta

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch

def load_model(model_name, checkpoint_path, in_channels, num_classes, device):
    if model_name == "unet":
        model = UNet(in_channels, num_classes)
    elif model_name == "deeplabv3plus":
        model = DeepLabv3Plus(in_channels, num_classes)
    elif model_name == "hrnet":
        model = HRNet(in_channels, num_classes)
    elif model_name == "segformer":
        model = SegFormer(in_channels, num_classes)
    elif model_name == "bisenetv1":
        model = BiSeNetV1(in_channels, num_classes)
    elif model_name == "bisenetv2":
        model = BiSeNetV2(in_channels, num_classes)
    elif model_name == "fast_scnn":
        model = Fast_SCNN(in_channels, num_classes)
    elif model_name == "seaformerpp":
        model = SeaFormerPP(in_channels, num_classes)
    elif model_name == "hrcloudnet":
        model = HRcloudNet(in_channels, num_classes)
    elif model_name == "cdnetv2":
        model = CDNetV2(in_channels, num_classes)
    elif model_name == "unetformer":
        model = UNetFormer(in_channels, num_classes)
    elif model_name == "cmlformer":
        model = CMLFormer(in_channels, num_classes)
    elif model_name == "cmtfnet":
        model = CMTFNet(in_channels, num_classes)
    elif model_name == "terrasegnet":
        model = TerraSegNet(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, dataloader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting -> {save_dir}"):
            if batch is None:
                continue
            for inputs, image_path, meta in batch:
                inputs = inputs.unsqueeze(0).to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                meta.update({"count": 1, "dtype": "uint8"})
                filename = os.path.basename(image_path)
                out_path = os.path.join(save_dir, filename)

                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(pred, 1)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'airpolsar'
    in_channels = 3
    num_classes = 6
    print(f"[INFO] Using device: {device}")

    test_dataset = SatelliteDataset(os.path.join("dataset", dataset, "test", "images"))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=safe_collate)

    model_info = {
        "unet": f"weights/{dataset}/unet.pth",
        "deeplabv3plus": f"weights/{dataset}/deeplabv3plus.pth",
        "hrnet": f"weights/{dataset}/hrnet.pth",
        "segformer": f"weights/{dataset}/segformer.pth",
        "bisenetv1": f"weights/{dataset}/bisenetv1.pth",
        "bisenetv2": f"weights/{dataset}/bisenetv2.pth",
        "fast_scnn": f"weights/{dataset}/fast_scnn.pth",
        "seaformerpp": f"weights/{dataset}/seaformerpp.pth",
        "hrcloudnet": f"weights/{dataset}/hrcloudnet.pth",
        "cdnetv2": f"weights/{dataset}/cdnetv2.pth",
        "unetformer": f"weights/{dataset}/unetformer.pth",
        "cmlformer": f"weights/{dataset}/cmlformer.pth",
        "cmtfnet": f"weights/{dataset}/cmtfnet.pth",
        "terrasegnet": f"weights/{dataset}/terrasegnet.pth",
    }

    for model_name, weights_path in model_info.items():
        print(f"\n[INFO] Processing model: {model_name}")
        model = load_model(model_name, weights_path, in_channels, num_classes, device)
        save_dir = os.path.join('output', dataset, model_name)
        predict_and_save(model, test_loader, device, save_dir)

    print("[INFO] Completed.")

if __name__ == "__main__":
    main()