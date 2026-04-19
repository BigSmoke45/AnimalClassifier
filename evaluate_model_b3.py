import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/animals")
IMG_SIZE = 300
BATCH_SIZE = 128

def load_model(model_path="best_model_b3.pth"):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(DEVICE)
    return model, classes

def get_predictions(model, classes):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds, classes):
    # Рахуємо матрицю вручну
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1

    # Малюємо
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Передбачено", fontsize=13)
    ax.set_ylabel("Реально", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15)

    # Числа всередині клітинок
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i][j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i][j]),
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("confusion_matrix_model_b3.png", dpi=150)
    plt.show()
    print("Збережено: confusion_matrix.png")

def print_per_class_metrics(labels, preds, classes):
    print("\n" + "="*50)
    print(f"{'Клас':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Кількість':>10}")
    print("="*50)

    for i, cls in enumerate(classes):
        tp = np.sum((preds == i) & (labels == i))
        fp = np.sum((preds == i) & (labels != i))
        fn = np.sum((preds != i) & (labels == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total     = np.sum(labels == i)

        print(f"{cls:<12} {precision*100:>9.1f}% {recall*100:>9.1f}% {f1*100:>9.1f}% {total:>10}")

    print("="*50)
    overall = np.sum(labels == preds) / len(labels) * 100
    print(f"{'Overall':.<12} {overall:>9.1f}%")

if __name__ == "__main__":
    print(f"Пристрій: {DEVICE}")
    model, classes = load_model()
    print(f"Класи: {classes}")
    print("Збираємо передбачення...")

    labels, preds = get_predictions(model, classes)
    print_per_class_metrics(labels, preds, classes)
    plot_confusion_matrix(labels, preds, classes)