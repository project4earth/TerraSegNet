import os
import torch
import numpy as np
import rasterio
import warnings
from PIL import Image
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
import torchvision.transforms.functional as TF
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchprofile.profile")
torch.manual_seed(42)

lower_percentile = [0.056, 0.056, 0.056, 0.056, -0.96]
upper_percentile = [2.744, 2.744, 2.744, 2.744, 0.96]

def minmax_normalization(image):
    num_channels = image.shape[0]
    normalized_image = np.zeros_like(image, dtype=np.float32)
    for c in range(num_channels):
        channel_data = image[c]
        normalized_channel = (channel_data - lower_percentile[c]) / (upper_percentile[c] - lower_percentile[c])
        normalized_channel = np.clip(normalized_channel, 0, 1)
        normalized_image[c] = normalized_channel
    return normalized_image

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, num_channels=5):
        self.num_channels = num_channels
        self.image_paths = []
        self.channel_dirs = [f"b{i}" for i in range(2, 2 + num_channels - 1)]
        self.root_dir = root_dir

        for channel_dir in self.channel_dirs:
            if not os.path.exists(os.path.join(self.root_dir, channel_dir)):
                raise RuntimeError(f"Channel folder {channel_dir} does not exist in {self.root_dir}")

        channel_folder = os.path.join(self.root_dir, self.channel_dirs[0])
        for img_name in sorted(os.listdir(channel_folder)):
            if img_name.endswith('.tif'):
                img_path = os.path.join(channel_folder, img_name)

                if os.path.exists(img_path):
                    self.image_paths.append(img_name)  
                else:
                    print(f"Warning: Label file {img_path} does not exist.")

    def load_channels(self, image_dir, img_name):
        channels = []
        for channel_dir in self.channel_dirs:
            channel_path = os.path.join(image_dir, channel_dir, img_name)
            try:
                with rasterio.open(channel_path) as src:
                    channel_data = src.read().astype(np.float32) * 0.0001
                    channels.append(channel_data)
            except Exception as e:
                print(f"Error loading channel {channel_path}: {e}")
                raise RuntimeError(f"Failed to load channel {channel_path}")
        return np.stack(channels, axis=0)  
   
    def compute_indices(self, image):
        red = image[2]
        nir = image[3]
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = (nir-red)/(nir+red+1e-6)  
        return ndvi

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image_dir = self.root_dir
        filename = os.path.splitext(os.path.basename(img_name))[0]

        try:
            image = self.load_channels(image_dir, img_name)
            ndvi = self.compute_indices(image)
            combined = np.concatenate((image, np.expand_dims(ndvi, axis=0)), axis=0)
            normalized_combined = minmax_normalization(combined)
            input_data = torch.tensor(normalized_combined, dtype=torch.float32)

            return input_data, filename

        except Exception as e:
            print(f"Error reading file {img_name}: {e}")
            raise RuntimeError(f"Failed to load sample from dataset at index {idx}")

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

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting -> {save_dir}"):
            if batch is None:
                continue
            for inputs, image_path in batch:
                if inputs.ndim == 4 and inputs.shape[0] == 5 and inputs.shape[1] == 1:
                    inputs = inputs.permute(1, 0, 2, 3)

                inputs = inputs.to(device, dtype=torch.float32)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                filename = os.path.basename(image_path)
                filename = os.path.splitext(filename)[0] + ".png"
                out_path = os.path.join(save_dir, filename)

                Image.fromarray(pred, mode="L").save(out_path)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'whus2cd'
    in_channels = 5
    num_classes = 2
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