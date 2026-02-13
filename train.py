import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np

# ============================

# Configuration

# ============================

CSV_PATH  = ‘HAM10000_metadata.csv’  # Path to metadata CSV
IMG_DIR_1 = ‘HAM10000_images_part_1’  # Path to images part 1
IMG_DIR_2 = ‘HAM10000_images_part_2’  # Path to images part 2
ORGANIZED_DIR = ‘HAM10000_organized’  # Output directory

BATCH_SIZE = 32
LR         = 0.001
NUM_EPOCHS = 15

# ============================

# Step 1: Organize images into class folders

# ============================

print(“Reading metadata…”)
df = pd.read_csv(CSV_PATH)
print(f”Total images: {len(df)}”)
print(f”Class distribution:\n{df[‘dx’].value_counts()}”)

# Split into train/val with stratification

train_df, val_df = train_test_split(
df,
test_size=0.2,
random_state=42,
stratify=df[‘dx’]
)

def copy_images(dataframe, split_name):
“”“Copy images to organized directory structure”””
for _, row in dataframe.iterrows():
image_id = row[‘image_id’]
label    = row[‘dx’]
filename = image_id + ‘.jpg’

```
    # Find image in part_1 or part_2
    src = None
    for img_dir in [IMG_DIR_1, IMG_DIR_2]:
        candidate = os.path.join(img_dir, filename)
        if os.path.exists(candidate):
            src = candidate
            break
    
    if src is None:
        continue
    
    # Create destination folder
    dst_folder = os.path.join(ORGANIZED_DIR, split_name, label)
    os.makedirs(dst_folder, exist_ok=True)
    
    dst = os.path.join(dst_folder, filename)
    shutil.copy2(src, dst)
```

if not os.path.exists(ORGANIZED_DIR):
print(”\nOrganizing images into class folders…”)
copy_images(train_df, ‘train’)
copy_images(val_df,   ‘val’)
print(“Done!”)
else:
print(“Images already organized.”)

# ============================

# Step 2: Setup data loaders

# ============================

TRAIN_DIR = os.path.join(ORGANIZED_DIR, ‘train’)
VAL_DIR   = os.path.join(ORGANIZED_DIR, ‘val’)

# Data augmentation for training

train_transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(15),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
])

# No augmentation for validation

val_transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

# ============================

# Step 3: Handle class imbalance

# ============================

class_counts  = np.array([len(os.listdir(os.path.join(TRAIN_DIR, c)))
for c in train_dataset.classes])
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(
sample_weights,
num_samples=len(sample_weights),
replacement=True
)

train_loader = DataLoader(
train_dataset,
batch_size=BATCH_SIZE,
sampler=sampler,
num_workers=2
)
val_loader = DataLoader(
val_dataset,
batch_size=BATCH_SIZE,
shuffle=False,
num_workers=2
)

# ============================

# Step 4: Setup model

# ============================

device = torch.device(“cuda” if torch.cuda.is_available() else “cpu”)
print(f”\nUsing device: {device}”)
print(f”Classes: {train_dataset.classes}”)
print(f”Train: {len(train_dataset)} | Val: {len(val_dataset)}”)

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================

# Step 5: Training functions

# ============================

def train_one_epoch():
model.train()
running_loss, correct, total = 0.0, 0, 0

```
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item() * images.size(0)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)

return running_loss / total, correct / total
```

def validate():
model.eval()
running_loss, correct, total = 0.0, 0, 0

```
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

return running_loss / total, correct / total
```

# ============================

# Step 6: Training loop

# ============================

print(”\nStarting training…”)
for epoch in range(NUM_EPOCHS):
train_loss, train_acc = train_one_epoch()
val_loss, val_acc     = validate()
print(f”Epoch {epoch+1}/{NUM_EPOCHS} | “
f”Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | “
f”Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}”)

# ============================

# Step 7: Save model

# ============================

SAVE_PATH = ‘ham10000_mobilenetv2.pth’
torch.save(model.state_dict(), SAVE_PATH)
print(f”\nModel saved: {SAVE_PATH}”)
