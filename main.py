import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
import random
from PIL import Image, ImageFilter

# Кастомні деградації
class AddNoise:
    def __init__(self, max_level=40):
        self.max_level = max_level
    def __call__(self, img):
        level = random.uniform(0, self.max_level)
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, level, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class AddBlur:
    def __init__(self, max_radius=4):
        self.max_radius = max_radius
    def __call__(self, img):
        radius = random.uniform(0, self.max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class RandomDarkness:
    def __init__(self, min_factor=0.3):
        self.min_factor = min_factor
    def __call__(self, img):
        factor = random.uniform(self.min_factor, 1.0)
        arr = np.array(img).astype(np.float32)
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class RandomLowRes:
    def __init__(self, min_size=32):
        self.min_size = min_size
    def __call__(self, img):
        if random.random() < 0.3:
            size = random.randint(self.min_size, 96)
            img = img.resize((size, size), Image.BILINEAR)
            img = img.resize((224, 224), Image.BILINEAR)
        return img

# Головна функція
def main():
    DATA_DIR = Path("data/animals")
    BATCH_SIZE = 128
    EPOCHS = 15
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Використовується: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # ── Нові деградації ──
        AddBlur(max_radius=4),
        AddNoise(max_level=40),
        RandomDarkness(min_factor=0.3),
        RandomLowRes(min_size=32),
        # ─────────────────────
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    classes = full_dataset.classes
    print(f"\nКласи: {classes}")
    print(f"Всього фото: {len(full_dataset)}")

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    val_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {train_size} | Val: {val_size}")
    print(f"Кількість по класах: {dict(zip(classes, class_counts))}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(x in name for x in ["features.6", "features.7", "features.8", "classifier"]):
            param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nПараметрів навчається: {trainable:,} з {total:,} ({100*trainable/total:.1f}%)")

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and "classifier" in n]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 1e-3},
    ])

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.3
    )

    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP = 5

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / train_size * 100
        val_acc = val_correct / val_size * 100
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "val_acc": best_val_acc,
                "epoch": epoch + 1,
            }, "best_model.pth")
            print(f"  ✅ Збережено! Best val acc: {best_val_acc:.1f}%")
        else:
            patience_counter += 1
            print(f"  ⏳ Без покращення {patience_counter}/{EARLY_STOP}")
            if patience_counter >= EARLY_STOP:
                print(f"\n🛑 Early stopping на epoch {epoch+1}")
                break

    print(f"\nГотово! Найкраща точність: {best_val_acc:.1f}%")

if __name__ == "__main__":
    main()