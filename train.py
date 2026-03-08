"""
DeepShield — train.py
Local PC pe train karo — CPU + GPU dono support
Dataset: 140k Real and Fake Faces (Kaggle)

HOW TO RUN:
  python train.py

Output:
  deepfake_model.pth  ← yeh file app.py ke saath rakhna
"""

import os, time, copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "dataset/real_vs_fake"
MODEL_SAVE   = "deepfake_model.pth"
EPOCHS       = 10
BATCH_SIZE   = 32    # RAM kam ho toh 16 karo
LR           = 0.0001
NUM_WORKERS  = 0     # Windows pe 0 rakhna

# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n" + "="*55)
print("  DeepShield Model Training")
print("="*55)
print(f"  Device  : {device}")
if device.type == 'cuda':
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs  : {EPOCHS} (GPU fast hai!)")
else:
    print(f"  CPU mode: slow hoga, ~30-60 min/epoch")
    print(f"  Tip     : Raat ko chalao aur so jao!")
print("="*55 + "\n")

# ─────────────────────────────────────────────
# DATASET CHECK
# ─────────────────────────────────────────────
train_path = os.path.join(DATASET_PATH, "train")
valid_path = os.path.join(DATASET_PATH, "valid")

if not os.path.exists(train_path):
    print(f"ERROR: Dataset nahi mila: '{train_path}'")
    print("Check karo dataset/real_vs_fake/train/ folder exist karta hai!")
    exit(1)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("Dataset load ho raha hai...")
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS,
                          pin_memory=True if device.type=='cuda' else False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

print(f"Classes: {train_dataset.class_to_idx}")
# Class order automatically detect karo
FAKE_LABEL = train_dataset.class_to_idx.get('fake', 1)
REAL_LABEL = train_dataset.class_to_idx.get('real', 0)
print(f"  → Fake label: {FAKE_LABEL} | Real label: {REAL_LABEL}")
print(f"Train images : {len(train_dataset):,}")
print(f"Valid images : {len(valid_dataset):,}\n")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Early layers freeze karo
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
        # Classifier replace karo
        n = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n, 2)
        )

    def forward(self, x):
        return self.model(x)

model = DeepfakeDetector().to(device)
print("Model ready!\n")

# ─────────────────────────────────────────────
# TRAINING SETUP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

# ─────────────────────────────────────────────
# TRAIN + EVAL FUNCTIONS
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        if (i + 1) % 100 == 0:
            acc = 100 * correct / total
            print(f"  [{i+1:4d}/{len(loader)}] "
                  f"Loss: {total_loss/(i+1):.4f} | "
                  f"Acc: {acc:.2f}%", end='\r')
    print()
    return total_loss / len(loader), 100 * correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
print("="*55)
print(f"  Training shuru — {EPOCHS} epochs")
print("="*55)

best_val_acc   = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
history = {'train_acc': [], 'val_acc': []}

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss,   val_acc   = eval_epoch(model, valid_loader, criterion)
    scheduler.step(val_acc)

    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    elapsed = time.time() - start
    mins    = int(elapsed // 60)
    secs    = int(elapsed % 60)

    print(f"  Train  — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"  Valid  — Loss: {val_loss:.4f}  | Acc: {val_acc:.2f}%")
    print(f"  Time   — {mins}m {secs}s")

    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, MODEL_SAVE)
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")
    else:
        print(f"  (Best so far: {best_val_acc:.2f}%)")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  Training Complete!")
print(f"  Best Validation Accuracy : {best_val_acc:.2f}%")
print(f"  Model saved              : {MODEL_SAVE}")
print("="*55)
print("\nNext Steps:")
print("  1. deepfake_model.pth DeepShield folder mein hai")
print("  2. python app.py chalao")
print("  3. Terminal mein dikhega: Trained weights loaded")
print("  4. Ab results accurate honge!\n")

print("Epoch-wise Summary:")
print(f"{'Epoch':<8} {'Train Acc':<12} {'Val Acc'}")
print("-" * 32)
for i, (ta, va) in enumerate(zip(history['train_acc'], history['val_acc']), 1):
    print(f"{i:<8} {ta:<12.2f} {va:.2f}%")
