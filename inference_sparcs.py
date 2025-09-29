import os
import torch
import numpy as np
from PIL import Image
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

class SatelliteDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []

        image_dir = root_dir
        for img_name in sorted(os.listdir(image_dir)):
            if img_name.endswith('.png'):
                img_path = os.path.join(image_dir, img_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                else:
                    print(f"Error: Image file {img_path} does not exist.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        filename = os.path.splitext(os.path.basename(img_path))[0]

        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = TF.to_tensor(image)

            return image_tensor, filename

        except Exception as e:
            print(f"Error reading file {img_path}: {e}")
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
                inputs = inputs.unsqueeze(0).to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                filename = os.path.basename(image_path)
                filename = os.path.splitext(filename)[0] + ".png"
                out_path = os.path.join(save_dir, filename)

                Image.fromarray(pred, mode="L").save(out_path)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'sparcs'
    in_channels = 3
    num_classes = 3
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