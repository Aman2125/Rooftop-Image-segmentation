import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# ========== CONFIG ==========
IMAGE_DIR = "images"
MASK_DIR = "combined_masks"
PRED_DIR = "predictions"
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PRED_DIR, exist_ok=True)

# ========== DATASET ==========
class SegDataset(Dataset):
    def __init__(self, img_files, mask_files):
        self.img_files = img_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB").resize((512, 512))
        mask = Image.open(self.mask_files[idx]).convert("L").resize((512, 512))

        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask

# ========== MODEL ==========
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            CBR(3, 64),
            CBR(64, 128),
            nn.MaxPool2d(2)
        )

        self.middle = CBR(128, 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            CBR(64, 64),
            nn.Conv2d(64, 3, 1)  # 3 classes: background, rooftop, obstruction
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# ========== UTILS ==========
def get_file_dict(directory, extensions=(".png", ".jpg", ".jpeg")):
    files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    return {os.path.splitext(f)[0]: os.path.join(directory, f) for f in files}

def to_one_channel_mask(pred):
    pred = pred.argmax(0).byte().cpu().numpy()
    result = np.zeros_like(pred, dtype=np.uint8)
    result[pred == 1] = 255       # rooftop
    result[pred == 2] = 127       # obstruction
    return result

# ========== LOAD DATA ==========
image_dict = get_file_dict(IMAGE_DIR)
mask_dict = get_file_dict(MASK_DIR)

common_keys = list(set(image_dict.keys()) & set(mask_dict.keys()))
common_keys.sort()

if len(common_keys) == 0:
    print("No matching images and masks found. Check filenames!")
    exit()

image_paths = [image_dict[k] for k in common_keys]
mask_paths = [mask_dict[k] for k in common_keys]

train_imgs, test_imgs, train_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
train_loader = DataLoader(SegDataset(train_imgs, train_masks), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SegDataset(test_imgs, test_masks), batch_size=1)

# ========== TRAIN ==========
model = UNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_iou = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # ========== VALIDATE ==========
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img_path, mask_path in zip(test_imgs, test_masks):
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)

            pred_mask = to_one_channel_mask(output.squeeze())
            gt_mask = np.array(Image.open(mask_path).resize((512, 512)))
            gt_mask[gt_mask == 255] = 1
            gt_mask[gt_mask == 127] = 2
            pred_mask[pred_mask == 255] = 1
            pred_mask[pred_mask == 127] = 2

            y_true.append(gt_mask.flatten())
            y_pred.append(pred_mask.flatten())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)

    acc = accuracy_score(y_true_np, y_pred_np)
    iou = jaccard_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    print(f"Validation -> Acc: {acc:.4f} | IoU: {iou:.4f}")

    # Save best model
    if iou > best_iou:
        best_iou = iou
        torch.save(model.state_dict(), "unet_best_model.pth")
        print(f"✅ Saved new best model at epoch {epoch+1} with IoU {iou:.4f}")

# Always save final
torch.save(model.state_dict(), "unet_final_model.pth")
print("🚀 Training complete. Final model saved.")
